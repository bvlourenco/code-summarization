import torch

from model.modules.collapse_copy_scores import collapse_copy_scores


def translate_tokens(tokens_idx, source_vocab, target_vocab):
    '''
    Given a list of token predictions where each token is a number, it translates
    each token from a number to its respective string.

    Args:
        tokens_idx: A list of numbers where each number is the index of a token
        vocabulary: The vocabulary from where the tokens belong (code vocabulary
                    or summaries vocabulary).

    Returns:
        A string with the textual representation of the tokens. Tokens are 
        separated with whitespaces.
    '''
    translation = ""
    # tokens_idx[1:] -> Ignoring the first token, which is <BOS>
    for token_idx in tokens_idx[1:]:

        token_idx_num = token_idx.item()

        # Because of copy attention, 
        if token_idx_num >= len(target_vocab.idx_to_token):
            token_idx_norm = token_idx_num - len(target_vocab.idx_to_token)
            if token_idx_norm in source_vocab.idx_to_token:
                token = source_vocab.idx_to_token[token_idx_norm]
            elif token_idx_norm in target_vocab.idx_to_token:
                token = target_vocab.idx_to_token[token_idx_norm]
            else:
                token = "<UNK>"
        else:
            token = target_vocab.idx_to_token[token_idx_num]
        
        if token in ["<EOS>", "<PAD>"]:
            break
        elif token == "<BOS>":
            raise ValueError("Unexpected <BOS>")
        else:
            translation += token + " "
    
    if translation == "":
        translation = "<PAD>"

    # Removing the last (extra) space from the translation
    return translation.strip()


def greedy_decode(model, 
                  src, 
                  device, 
                  start_symbol_idx, 
                  end_symbol_idx, 
                  max_tgt_len,
                  copy_attn,
                  src_map,
                  source_vocab,
                  target_vocab):
    '''
    Creates a code comment given a code snippet using the greedy decoding
    strategy (where we generate one token at a time by greedily selecting the
    token with the highest probability at each step)

    Args:
        model: The model (an instance of Transformer). 
        src: The input (a code snippet numericalized). Shape: `(batch_size, src_len)` 
        device: The device where the model and tensors are inserted (GPU or CPU). 
        start_symbol_idx: The vocabulary start symbol index (<BOS> index)
        end_symbol_idx: The vocabulary end symbol index (<EOS> index)
        max_tgt_len (int): Maximum length of the summary.
        TODO: FINISH ARGS!

    Returns:
        tgt: The predicted code comment token indexes. Shape: `(1, tgt_length)`

    Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
    '''
    src_mask = model.generate_src_mask(src)
    enc_output = model.encode(src, src_mask).to(device)

    # Initializes the target sequence with <BOS> symbol
    # Will be further expland to include the predicted tokens
    tgt = torch.ones(1, 1).fill_(start_symbol_idx).type(torch.long).to(device)

    for _ in range(max_tgt_len - 2):

        if copy_attn:
            # Turn any copied words into UNKs.
            tgt = tgt.masked_fill(
                tgt.gt(len(target_vocab) - 1), target_vocab.token_to_idx['<UNK>']
            )
            
        # Generate probabilities for the next token
        # Essentially, we're running the decoder phasing of the model
        # to generate the next token
        tgt_mask = model.generate_tgt_mask(tgt)
        dec_output, dec_cross_attn_score = model.decode(src, src_mask, tgt, tgt_mask, enc_output)

        if copy_attn:
            # Getting the attention of the last head and only for the last target token
            attn_last_head = dec_cross_attn_score[:, -1, -1, :]
            # attn = attn_last_head.contiguous().view(-1, attn_last_head.shape[-1])
            # dec_output[:, -1] -> Selects only the output for the last token
            # hidden = dec_output[:, -1].contiguous().view(-1, dec_output.shape[-1])
            prob = model.fc(dec_output[:, -1], attn_last_head, src_map)

            batch_size = 1
            prob = prob.view(-1, batch_size, prob.size(-1))
            prob = prob.transpose(0, 1).contiguous()
            prob = collapse_copy_scores(prob,
                                        src,
                                        source_vocab,
                                        target_vocab,
                                        source_vocab.token_to_idx['<UNK>'],
                                        source_vocab.token_to_idx['<BOS>'],
                                        source_vocab.token_to_idx['<EOS>'],
                                        source_vocab.token_to_idx['<PAD>'],
                                        batch_size,
                                        device)
            prob = prob.view(-1, prob.size(-1))
        else:
            # prob has shape: `(batch_size, tgt_vocab_size)`
            prob = model.fc(dec_output[:, -1])

        # Get the index of of the token with the
        # highest probability as the predicted next word.
        _, next_word = torch.max(prob, dim=1)

        # Get the integer value present in the tensor. It represents
        # the index of the predicted token.
        next_word = next_word.item()

        if next_word > 30000:
            next_word = 3

        # Adding the new predicted token to tgt
        tgt = torch.cat([tgt,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

        # If the token predicted is <EOS>, then stop generating new tokens.
        if next_word == end_symbol_idx:
            break

    # Adding <EOS> token
    if tgt[:, -1] != end_symbol_idx:
        tgt = torch.cat([tgt,
                        torch.ones(1, 1).type_as(src.data).fill_(end_symbol_idx)], dim=1)

    # Return the indexes of the predicted tokens
    return tgt


def beam_search(model, src, device, start_symbol_idx, end_symbol_idx, max_tgt_len, beam_size):
    '''
    Creates a code comment given a code snippet using the beam search strategy
    (at each timestep, keeps the k most probable partially decoded sequences and 
    stops when all these sequences generate the <EOS> token or the `max_tgt_length`
    is reached)

    Args:
        model: The model (an instance of Transformer). 
        src: The input (a code snippet numericalized). Shape: `(batch_size, src_len)` 
        device: The device where the model and tensors are inserted (GPU or CPU). 
        start_symbol_idx: The vocabulary start symbol index (<BOS> index)
        end_symbol_idx: The vocabulary end symbol index (<EOS> index)
        max_tgt_len (int): Maximum length of the summary.
        beam_size (int): Number of elements to store during beam search
                         Only applicable if `mode == 'beam'`

    Returns:
        tgt: The predicted code comment token indexes. Shape: `(1, tgt_length)`
    '''
    src_mask = model.generate_src_mask(src)
    enc_output = model.encode(src, src_mask).to(device)

    # Initialize the beam with the start symbol
    beam = [(torch.ones(1, 1).fill_(
        start_symbol_idx).type(torch.long).to(device), 0)]

    # Generate tokens using beam search
    for _ in range(max_tgt_len - 2):
        new_beam = []

        # Expand each beam
        for seq, score in beam:
            # If the sequence ends with the end symbol, add it to the new beam as it is
            if seq[-1].item() == end_symbol_idx:
                new_beam.append((seq, score))
            else:
                # Generate probabilities for the next token
                tgt_mask = model.generate_tgt_mask(seq)
                dec_output = model.decode(
                    src, src_mask, seq, tgt_mask, enc_output)
                prob = model.fc(dec_output[-1, :, :])

                # Get the top-k most probable tokens
                topk_probs, topk_indices = torch.topk(prob, beam_size, dim=1)

                # Extend the sequences with the top-k tokens and update their scores
                for i in range(beam_size):
                    new_seq = torch.cat(
                        (seq, topk_indices[:, i].unsqueeze(1)), dim=0)
                    new_score = score + topk_probs[:, i].item()
                    new_beam.append((new_seq, new_score))

        # Select the top-k beams with the highest scores
        new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[
            :beam_size]

        # Check if all beams have reached the end symbol
        if all(seq[-1].item() == end_symbol_idx for seq, _ in new_beam):
            break

        beam = new_beam

    # Getting the sequence with highest score
    best_sequence, _ = beam[0]

    # Adding <EOS> token
    best_sequence = torch.cat([best_sequence,
                               torch.ones(1, 1).type_as(src.data).fill_(end_symbol_idx)], 
                               dim=0)

    return best_sequence

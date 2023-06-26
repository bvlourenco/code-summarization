import torch


def translate_tokens(tokens_idx, vocabulary):
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
        # Token not in vocabulary -> Unknown token
        if token_idx.item() not in vocabulary.idx_to_token:
            token = "<UNK>"
        else:
            token = vocabulary.idx_to_token[token_idx.item()]
        
        if token in ["<EOS>", "<PAD>"]:
            break
        elif token == "<BOS>":
            raise ValueError("Unexpected <BOS>")
        elif token == "<UNK>":
            token = "?"
        translation += token + " "
    
    if translation == "":
        translation = "<PAD>"

    # Removing the last (extra) space from the translation
    return translation.strip()


def greedy_decode(model, src, device, start_symbol_idx, end_symbol_idx, max_tgt_len):
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
        # Generate probabilities for the next token
        # Essentially, we're running the decoder phasing of the model
        # to generate the next token
        tgt_mask = model.generate_tgt_mask(tgt)
        dec_output = model.decode(src, src_mask, tgt, tgt_mask, enc_output)
        # prob has shape: `(batch_size, tgt_vocab_size)`
        prob = model.fc(dec_output[:, -1])

        # Get the index of of the token with the
        # highest probability as the predicted next word.
        _, next_word = torch.max(prob, dim=1)

        # Get the integer value present in the tensor. It represents
        # the index of the predicted token.
        next_word = next_word.item()

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
                               torch.ones(1, 1).type_as(src.data).fill_(end_symbol_idx)], dim=0)

    return best_sequence

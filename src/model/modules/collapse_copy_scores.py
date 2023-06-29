import torch


def collapse_copy_scores(scores,
                         source_vocab,
                         tgt_vocab,
                         unk_idx,
                         batch_size,
                         device
                         ):
    """
    Given scores from an expanded dictionary corresponeding to a batch, sums 
    together copies, with a dictionary word when it is ambiguous.

    Takes into account ambiguous cases where a source word can correspond to 
    multiple target words.

    Source: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/copy_generator.py
    """

    # Represents the starting index of the extended vocabulary
    # (including both target and source words) in the `scores`` tensor.
    offset = len(tgt_vocab)

    for b in range(batch_size):
        # Store the indices of ambiguous source words
        blank = []

        # Store the target word indices corresponding to the ambiguous source words
        fill = []

        # Iterate over the source vocabulary, passing through all indices
        # If the index corresponds to the <UNK> token, we ignore.
        # Maps each source word sw to its corresponding target index ti
        # using the tgt_vocab.
        for i in range(len(source_vocab)):
            if i == unk_idx:
                continue

            sw = source_vocab.idx_to_token[i]

            if sw in tgt_vocab.token_to_idx:
                ti = tgt_vocab.token_to_idx[sw]
            else:
                ti = unk_idx

            # If the target index is non-zero (indicating it is a known target
            # word), the source index (offset + i) is added to blank and the
            # target index ti is added to fill.
            if ti != unk_idx:
                blank.append(offset + i)
                fill.append(ti)

        # If there are any ambiguous mappings (i.e., blank is not empty)
        if blank:
            blank = torch.tensor(blank, dtype=torch.int64, device=device)
            fill = torch.tensor(fill, dtype=torch.int64, device=device)
            # Retrieves the scores for the current example from the scores tensor
            score = scores[b]

            # TODO: Check if index_add_ and index_select and index_fill_ have the
            # first argument correct or not.

            # Adds the scores of the ambiguous source words to the corresponding
            # target words. Selects the columns specified by `blank` indices and
            # adds them to the columns specified by fill indices.
            score.index_add_(1, fill, score.index_select(1, blank))
            # Fills the columns specified by the `blank` indices in the score
            # tensor with a small value (1e-10 in this case).
            score.index_fill_(1, blank, 1e-10)

            scores[b] = score
    return scores

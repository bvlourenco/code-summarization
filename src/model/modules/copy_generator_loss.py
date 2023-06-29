import torch
import torch.nn as nn


class CopyGeneratorLoss(nn.Module):
    """
    Copy generator criterion. It is used with the copy generation mechanism.

    Source: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/copy_generator.py
    """

    def __init__(self, 
                 vocab_size, 
                 force_copy, 
                 unk_index, 
                 ignore_index,
                 eps=1e-20
                ):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, scores, align, target):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            align (LongTensor): ``(batch_size x tgt_len)``
                                Represents the alignment between target words and 
                                source words.
            target (LongTensor): ``(batch_size x tgt_len)``
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # Has the indices corresponding to the copied tokens in the scores tensor.
        copy_ix = align.unsqueeze(1) + self.vocab_size

        # Has the probability of the tokens copied from source.
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)

        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            # The non-copy variable is further updated to include positions 
            # where target is not equal to self.unk_index as well.
            non_copy = non_copy | (target != self.unk_index)

        # Combines the probabilities of copied tokens and probabilities of 
        # generating tokens from the vocabulary. 
        # If non_copy is True -> Select the elements from copy_tok_probs + vocab_probs
        # if non_copy is False -> Select elements from copy_tok_probs
        probs = torch.where(non_copy, 
                            copy_tok_probs + vocab_probs, 
                            copy_tok_probs)

        # computing the NLLLoss
        loss = -probs.log()
        return loss

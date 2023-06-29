import torch
import torch.nn as nn


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.

    These networks consider copying words
    directly from the source sequence.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`

    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int): Index of the padding token in the source vocabulary.

    Source: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/copy_generator.py
    """

    def __init__(self, input_size, output_size, pad_idx):
        super(CopyGenerator, self).__init__()
        # Generates the standard softmax probabilities over the target vocabulary
        self.linear = nn.Linear(input_size, output_size)

        # Compute the probability of copying a word from the source
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           hidden (FloatTensor): hidden output ``(batch x tlen, input_size)``
                                 Hidden output of the encoder or decoder.
           attn (FloatTensor): attn for each ``(batch x tlen, slen)``
                               Attention distribution over the source sequence.
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab.
               ``(batch, src_len, extra_words)``
               For each source sentence we have a `src_map` that maps
               each source word to an index in `tgt_dict` if it known, or
               else to an extra word.
        """
        _, slen = attn.size()
        batch, _, cvocab = src_map.size()

        # Original probabilities.
        logits = self.linear(hidden)
        # Probabilities set to -inf to exclude them from consideration
        logits[:, self.pad_idx] = -float("inf")
        prob = torch.softmax(logits, 1)

        # Probability of copying a word ( p(z=1) batch ).
        p_copy = torch.sigmoid(self.linear_copy(hidden))

        # Probability of not copying a word: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)

        # Modified attention weights that reflect the probability of coping a word
        # from the source code.
        mul_attn = torch.mul(attn, p_copy)

        # Computes the copy probabilities for each target word by aggregating
        # the attention weights over the source sequence.
        # Represents the probabilities of copying each word from the source
        copy_prob = torch.bmm(mul_attn.view(-1, batch, slen).transpose(0, 1), src_map)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)

        # Represents the distribution over the target dictionary extended by
        # the dynamic dictionary implied by copying source words.
        # Combines the probabilities of generating words from the target
        # vocabulary (out_prob) and copying words from the source sequence
        # (copy_prob).
        return torch.cat([out_prob, copy_prob], 1)

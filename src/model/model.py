from datetime import datetime
import json
import logging
import sys
import traceback
from nltk.metrics.scores import precision, recall, f_measure
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from evaluation.neuralcodesum.eval import eval_accuracies
from evaluation.graphs import display_attention
from evaluation.translation import beam_search, greedy_decode, translate_tokens
from model.transformer.transformer_model import Transformer
from rouge_score import rouge_scorer
from torch.nn.parallel import DistributedDataParallel as DDP

# Multiple calls to getLogger() with the same name will return a reference
# to the same Logger object, which saves us from passing the logger objects
# to every part where it’s needed.
# Source: https://realpython.com/python-logging/
logger = logging.getLogger('main_logger')
# To avoid having repeated logs!
logger.propagate = False


class Model:
    '''
    Class that represents a model and common operations to them (training, 
    evaluation and other).
    '''

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model,
                 num_heads,
                 num_layers,
                 d_ff,
                 max_src_length,
                 max_tgt_length,
                 dropout,
                 learning_rate,
                 label_smoothing,
                 pad_idx,
                 device,
                 gpu_rank,
                 init_type,
                 optimizer,
                 hyperparameter_hsva,
                 hyperparameter_data_flow,
                 hyperparameter_control_flow,
                 hyperparameter_ast):
        '''
        Args:
            src_vocab_size (int): size of the source vocabulary.
            tgt_vocab_size (int): size of the target vocabulary.
            d_model (int): dimensionality of the model.
            num_heads (int): number of heads of the Multi Head Attention.
            num_layers (int): number of encoder and decoder layers.
            d_ff (int): the hidden layer size of the second-layer of the Feed
                        Forward Network (in encoder and decoder).
            max_src_length (int): maximum length of the source code.
            max_tgt_length (int): maximum length of the summaries.
            dropout (int): dropout probability (between 0 and 1).
            learning_rate (int): Value of the initial learning rate.
            label_smoothing (int): Value of label smoothing to be applied in 
                                   loss function.
            pad_idx (int): index of the <PAD> token
            device: The device where the model and tensors are inserted (GPU or CPU).
            gpu_rank (int): The rank of the GPU.
                            It has the value of None if no GPUs are avaiable or
                            only 1 GPU is available.
            init_type (string): The weight initialization technique to be used
                                with the Transformer architecture.
            hyperparameter_hsva (int): Hyperparameter used in HSVA (Hierarchical
                                       Structure Variant Attention) to control the
                                       distribution of the heads by type.
            hyperparameter_data_flow (int): Hyperparameter used to adjust the 
                                            weight of the data flow adjacency 
                                            matrix in the self-attention.
            hyperparameter_control_flow (int): Hyperparameter used to adjust the 
                                               weight of the control flow adjacency 
                                               matrix in the self-attention.
            hyperparameter_ast (int): Hyperparameter used to adjust the 
                                      weight of the ast adjacency matrix in the 
                                      self-attention.
        '''
        self.model = Transformer(src_vocab_size,
                                 tgt_vocab_size,
                                 d_model,
                                 num_heads,
                                 num_layers,
                                 d_ff,
                                 max_src_length,
                                 max_tgt_length,
                                 dropout,
                                 device,
                                 init_type,
                                 hyperparameter_hsva,
                                 hyperparameter_data_flow,
                                 hyperparameter_control_flow,
                                 hyperparameter_ast)

        self.num_heads = num_heads

        # Passing the model and all its layers to GPU if available
        self.model = self.model.to(device)

        if gpu_rank is not None:
            # Used to run model in several GPUs
            self.model = DDP(self.model,
                             device_ids=[gpu_rank],
                             output_device=gpu_rank)

        # Function used to compute the loss. ignore_index is the padding token index
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx,
                                             label_smoothing=label_smoothing)

        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=learning_rate,
                                        betas=(0.9, 0.98),
                                        eps=1e-9)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=learning_rate)

        # Adjusts the learning rate during training
        # Implements the "Noam" learning rate scheduler, used in vanilla Transformer
        # self.scheduler = LambdaLR(
        #     optimizer=self.optimizer,
        #     lr_lambda=lambda step: Model.rate(
        #         step, model_size=d_model, factor=1.0, warmup=400
        #     ),
        # )

        self.device = device
        self.gpu_rank = gpu_rank

        self.tgt_vocab_size = tgt_vocab_size
        self.max_tgt_length = max_tgt_length

    @staticmethod
    def rate(step, model_size, factor, warmup):
        '''
        Compute the learning rate at each step based on the model size, 
        factor, and warmup.

        Args:
            step: Current step in the training process.
            model_size: Size of the model, typically the dimensionality of the 
                        model (d_model).
            factor: Scaling factor for the learning rate.
            warmup: Number of warmup steps for gradual learning rate increase.

        Returns:
            The computed learning rate for the given step.

        Note: We have to default the step to 1 for LambdaLR function to avoid 
              zero raising to negative power.

        Source: https://nlp.seas.harvard.edu/annotated-transformer/
        '''
        if step == 0:
            step = 1
        return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
        )

    def train_epoch(self, train_dataloader, grad_clipping, display_attn=False):
        '''
        Trains the model during one epoch, using the examples of the dataset for training.
        Computes the loss during this epoch.

        Args:
            train_dataloader: A Dataloader object that contains the training set.
            grad_clipping (int): Maximum norm of the gradient.
            display_attn (bool): Tells whether we want to save the attention of
                                 the last layer encoder and decoder 
                                 multi-head attentions.

        Returns:
            The average loss across all examples during one epoch.

        Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
                https://medium.com/@hunter-j-phillips/putting-it-all-together-the-implemented-transformer-bfb11ac1ddfe
        '''
        # Set the model to training mode
        self.model.train()
        losses = 0

        for src_tokens, tgt_tokens, src, tgt, source_ids, source_mask, token, \
                statement, data_flow, control_flow, ast in \
                                        tqdm(train_dataloader, desc="Training"):
            # Passing vectors to GPU if it's available
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            source_ids = source_ids.to(self.device)
            source_mask = source_mask.to(self.device)
            token = token.to(self.device)
            statement = statement.to(self.device)
            data_flow = data_flow.to(self.device)
            control_flow = control_flow.to(self.device)
            ast = ast.to(self.device)

            # Zeroing the gradients of transformer parameters
            self.optimizer.zero_grad()

            # Slicing the last element of tgt_data because it is used to compute the
            # loss.
            output, enc_attn, dec_self_attn, dec_cross_attn = self.model(src,
                                                                         source_ids,
                                                                         source_mask,
                                                                         tgt[:, :-1],
                                                                         token,
                                                                         statement,
                                                                         data_flow,
                                                                         control_flow,
                                                                         ast)

            if display_attn:
                self.display_all_attentions(src_tokens,
                                            tgt_tokens,
                                            enc_attn,
                                            dec_self_attn,
                                            dec_cross_attn)

            # - output is reshaped using view() method in shape (batch_size * tgt_len, tgt_vocab_size)
            # - tgt_data is reshaped using view() method in shape (batch_size * tgt_len)
            # - criterion(prediction_labels, target_labels)
            # - tgt[:, 1:] removes the first token from the target sequence since we are training the
            # model to predict the next word based on previous words.
            loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size),
                                  tgt[:, 1:].contiguous().view(-1))

            # Computing gradients of loss through backpropagation
            loss.backward()

            # clip the weights
            clip_grad_norm_(self.model.parameters(), grad_clipping)

            # Update the model's parameters based on the computed gradients
            self.optimizer.step()

            # Adjust learning rate
            # self.scheduler.step()

            losses += loss.item()

        return losses / len(train_dataloader)

    def validate(self,
                 val_dataloader,
                 mode,
                 target_vocab,
                 num_epoch,
                 beam_size,
                 display_attn=False):
        '''
        Validates the model after a training epoch to see how well he generalizes. 
        Does not update the weights of the model.

        Args:
            val_dataloader: A Dataloader object that contains the validation set.
            mode (string): Indicates whether we only want to compute validation loss or if we also
                           want to translate the source sentences in the validation set
                           (either using greedy decoding or beam search).
                           Can be one of the following: "loss", "greedy" or "beam"
            target_vocab: The vocabulary built from the summaries in training set.
            num_epoch (int): The current training epoch.
            beam_size (int): Number of elements to store during beam search
                             Only applicable if `mode == 'beam'`
            display_attn (bool): Tells whether we want to save the attention of
                                 the last layer encoder and decoder 
                                 multi-head attentions.

        Returns:
            The average loss across all examples during one epoch.
            The average BLEU metric if mode is "greedy" or "beam".

        Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
        '''
        # Set the model to evaluation mode
        self.model.eval()
        losses = 0

        # Used to compute the ROUGE-L score
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # Writing the translation results to a file
        if mode in ['beam', 'greedy']:
            log = open('../results/validation_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") +
                       '_gpu_' + str(self.gpu_rank) + '_epoch' + str(num_epoch) + '.json', 'w')
            metrics = {"number_examples": 0,
                       "sum_bleu": 0, "sum_meteor": 0, "sum_rouge_l": 0,
                       "sum_precision": 0, "sum_recall": 0, "sum_f1": 0}
            hypotheses = {}
            references = {}

        # evaluate without updating gradients
        # Tells pytorch to not calculate the gradients
        with torch.no_grad():
            for batch_idx, (code_tokens, summary_tokens, code, summary, src, tgt, \
                source_ids, source_mask, token, statement, data_flow, \
                control_flow, ast) in enumerate(tqdm(val_dataloader, desc="Validating")):
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                source_ids = source_ids.to(self.device)
                source_mask = source_mask.to(self.device)
                token = token.to(self.device)
                statement = statement.to(self.device)
                data_flow = data_flow.to(self.device)
                control_flow = control_flow.to(self.device)
                ast = ast.to(self.device)

                if mode in ['beam', 'greedy']:
                    metrics, hyps, refs = self.translate_evaluate(src,
                                                                  source_ids,
                                                                  source_mask,
                                                                  target_vocab,
                                                                  token,
                                                                  statement,
                                                                  data_flow,
                                                                  control_flow,
                                                                  ast,
                                                                  code,
                                                                  summary,
                                                                  log,
                                                                  scorer,
                                                                  mode,
                                                                  beam_size,
                                                                  metrics,
                                                                  batch_idx)
                    hypotheses.update(hyps)
                    references.update(refs)

                # Slicing the last element of tgt_data because it is used to
                # compute the loss.
                output, enc_attn, dec_self_attn, dec_cross_attn = self.model(src,
                                                                             source_ids,
                                                                             source_mask,
                                                                             tgt[:,:-1],
                                                                             token,
                                                                             statement,
                                                                             data_flow,
                                                                             control_flow,
                                                                             ast)

                if display_attn:
                    self.display_all_attentions(code_tokens,
                                                summary_tokens,
                                                enc_attn,
                                                dec_self_attn,
                                                dec_cross_attn)

                # - output is reshaped using view() method in shape (batch_size * tgt_len, tgt_vocab_size)
                # - tgt_data is reshaped using view() method in shape (batch_size * tgt_len)
                # - criterion(prediction_labels, target_labels)
                # - tgt[:, 1:] removes the first token from the target sequence since we are training the
                # model to predict the next word based on previous words.
                loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size),
                                      tgt[:, 1:].contiguous().view(-1))

                losses += loss.item()

        if mode in ['beam', 'greedy']:
            log.close()
            metrics["BLEU_N"], metrics["ROUGE-L_N"], metrics["METEOR_N"] = eval_accuracies(hypotheses, references)
            Model.compute_avg_metrics(metrics)
            return losses / len(val_dataloader), \
                   (metrics["sum_bleu"] / metrics["number_examples"]) * 100
        else:
            return losses / len(val_dataloader), None

    def test(self,
             test_dataloader,
             target_vocab,
             mode,
             beam_size):
        '''
        Tests the model with the testing set.

        Args:
            test_dataloader: A Dataloader object that contains the testing set.
            target_vocab: The vocabulary built from the summaries in training set.
            mode (string): Indicates what is the strategy to be used in translation.
                           It can either be a greedy decoding strategy or a beam 
                           search strategy.
                           Can be one of the following: "greedy" or "beam"
            beam_size (int): Number of elements to store during beam search
                             Only applicable if `mode == 'beam'`
        '''
        # Set the model to evaluation mode
        self.model.eval()

        # Used to compute the ROUGE-L score
        # RougeScorer has the line "logging.info("Using default tokenizer.")"
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        metrics = {"number_examples": 0,
                   "sum_bleu": 0, "sum_meteor": 0, "sum_rouge_l": 0,
                   "sum_precision": 0, "sum_recall": 0, "sum_f1": 0}
        
        hypotheses = {}
        references = {}

        # evaluate without updating gradients
        # Tells pytorch to not calculate the gradients
        with torch.no_grad():
            with open('../results/test_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") +
                      '_gpu_' + str(self.gpu_rank) + '.json', 'w') as log:
                for batch_idx, (code_tokens, summary_tokens, code, summary, src, tgt, \
                    source_ids, source_mask, token, statement, data_flow, \
                    control_flow, ast) in enumerate(tqdm(test_dataloader, desc="Testing")):
                    src = src.to(self.device)
                    tgt = tgt.to(self.device)
                    source_ids = source_ids.to(self.device)
                    source_mask = source_mask.to(self.device)
                    token = token.to(self.device)
                    statement = statement.to(self.device)
                    data_flow = data_flow.to(self.device)
                    control_flow = control_flow.to(self.device)
                    ast = ast.to(self.device)

                    metrics, hyps, refs = self.translate_evaluate(src,
                                                                  source_ids,
                                                                  source_mask,
                                                                  target_vocab,
                                                                  token,
                                                                  statement,
                                                                  data_flow,
                                                                  control_flow,
                                                                  ast,
                                                                  code,
                                                                  summary,
                                                                  log,
                                                                  scorer,
                                                                  mode,
                                                                  beam_size,
                                                                  metrics,
                                                                  batch_idx)
                    hypotheses.update(hyps)
                    references.update(refs)

        metrics["BLEU_N"], metrics["ROUGE-L_N"], metrics["METEOR_N"] = eval_accuracies(hypotheses, references)
        Model.compute_avg_metrics(metrics)

    def display_all_attentions(self,
                               src_tokens,
                               tgt_tokens,
                               enc_attn,
                               dec_self_attn,
                               dec_cross_attn):
        '''
        Display the encoder self-attention, decoder self-attention and decoder
        cross-attention.

        Args:
            src_tokens: The tokens of the source sequence.
                        Shape: `(batch_size, max_src_len)`
            tgt_tokens: The tokens of the target sequences.
                        Shape: `(batch_size, max_tgt_len)`
            enc_attn: The encoder self attention.
                      Shape: `(batch_size, num_heads, max_src_len, max_src_len)`
            dec_self_attn: The decoder self attention.
                           Shape: `(batch_size, num_heads, max_tgt_len, max_tgt_len)`
            dec_cross_attn: The decoder cross attention.
                            Shape: `(batch_size, num_heads, max_tgt_len, max_src_len)`
        '''
        # Displaying self and cross attention scores of the last encoder
        # and decoder layer for the first example passed as input
        display_attention(src_tokens[0],
                          src_tokens[0],
                          enc_attn[0],
                          "encoder_self_attn",
                          self.num_heads)

        display_attention(tgt_tokens[0],
                          tgt_tokens[0],
                          dec_self_attn[0],
                          "decoder_self_attn",
                          self.num_heads)

        display_attention(src_tokens[0],
                          tgt_tokens[0],
                          dec_cross_attn[0],
                          "decoder_cross_attn",
                          self.num_heads)

    def translate_evaluate(self,
                           src,
                           source_ids,
                           source_mask,
                           target_vocab,
                           token,
                           statement,
                           data_flow,
                           control_flow,
                           ast,
                           code,
                           summary,
                           log,
                           scorer,
                           mode,
                           beam_size,
                           metrics,
                           batch_idx):
        '''
        Translates a code snippet (giving its respective summary) and also
        computes the evaluation metrics for the sentence and adds it to the
        sum of evaluation metrics (present in `metrics` dictionary).

        Args:
            src: code snippet numericalized
            target_vocab: The vocabulary built from the summaries in training set.
            token: The token adjacency matrices. 
                   Shape: `(batch_size, max_src_len, max_src_len)`
            statement: The statement adjacency matrices. 
                       Shape: `(batch_size, max_src_len, max_src_len)`
            data_flow: The data flow adjacency matrices. 
                       Shape: `(batch_size, max_src_len, max_src_len)`
            control_flow: The control flow adjacency matrices. 
                          Shape: `(batch_size, max_src_len, max_src_len)`
            ast: The ast adjacency matrices. 
                 Shape: `(batch_size, max_src_len, max_src_len)`
            code (string): code snippet in textual form.
            summary (string): reference code comment.
            log: file to write the evaluation metrics of the generated summaries.
            scorer: A RougeScorer object to compute ROUGE-L.
            mode (string): Indicates whether we only want to compute validation 
                           loss or if we also want to translate the source 
                           sentences in the validation set (either using greedy 
                           decoding or beam search).
                           Can be one of the following: "loss", "greedy" or "beam"
            beam_size (int): Number of elements to store during beam search
                             Only applicable if `mode == 'beam'`
            metrics: A dictionary containing the number of pairs 
                     <code snippet, summary> and the sum for each evaluation 
                     metric.
            batch_idx (int): The batch number, indicating how many batches were
                             already processed before. Used to give a unique ID
                             for each predicted and reference summary. 
        '''
        batch_size = src.shape[0]

        bleu, meteor, rouge_l, \
            precision, recall, f1, hyps, refs = self.translate_sentence(src,
                                                                        source_ids,
                                                                        source_mask,
                                                                        target_vocab,
                                                                        token,
                                                                        statement,
                                                                        data_flow,
                                                                        control_flow,
                                                                        ast,
                                                                        code,
                                                                        summary,
                                                                        log,
                                                                        scorer,
                                                                        mode,
                                                                        beam_size,
                                                                        batch_idx)

        # Performing weighted average of metrics
        metrics["sum_bleu"] += bleu * batch_size
        metrics["sum_meteor"] += meteor * batch_size
        metrics["sum_rouge_l"] += rouge_l * batch_size
        metrics["sum_precision"] += precision * batch_size
        metrics["sum_recall"] += recall * batch_size
        metrics["sum_f1"] += f1 * batch_size

        metrics["number_examples"] += batch_size

        return metrics, hyps, refs

    @staticmethod
    def compute_avg_metrics(metrics):
        '''
        Given the sum of each evaluation metric (BLEU, METEOR, ROUGE-L, Precision,
        Recall and F1-score), it computes the average for each metric and prints
        it to the logger.

        Also prints the BLEU, METEOR and ROUGE-L used by NeuralCodeSum and 
        other related works.

        Args:
            metrics: A dictionary containing the number of pairs 
                     <code snippet, summary> and the sum for each evaluation 
                     metric. 
        '''
        final_bleu = (metrics["sum_bleu"] / metrics["number_examples"]) * 100
        final_meteor = (metrics["sum_meteor"] /
                        metrics["number_examples"]) * 100
        final_rouge_l = (metrics["sum_rouge_l"] /
                         metrics["number_examples"]) * 100
        final_precision = (metrics["sum_precision"] /
                           metrics["number_examples"]) * 100
        final_recall = (metrics["sum_recall"] /
                        metrics["number_examples"]) * 100
        final_f1 = (metrics["sum_f1"] / metrics["number_examples"]) * 100

        final_bleu_n = metrics["BLEU_N"]
        final_rouge_l_n = metrics["ROUGE-L_N"]
        final_meteor_n = metrics["METEOR_N"]

        logger.info(f"Metrics:\nBLEU: {final_bleu:7.3f} | " +
                    f"METEOR: {final_meteor:7.3f} | " +
                    f"ROUGE-L: {final_rouge_l:7.3f} | " +
                    f"Precision: {final_precision:7.3f} | " +
                    f"Recall: {final_recall:7.3f} | " +
                    f"F1: {final_f1:7.3f}")
        
        logger.info(f"NeuralCodeSum metrics:\nBLEU_N: {final_bleu_n:7.3f} | " +
                    f"METEOR_N: {final_meteor_n:7.3f} | " +
                    f"ROUGE-L_N: {final_rouge_l_n:7.3f}")

    def translate_sentence(self,
                           src,
                           source_ids,
                           source_mask,
                           target_vocab,
                           token,
                           statement,
                           data_flow,
                           control_flow,
                           ast,
                           code,
                           summary,
                           log,
                           scorer,
                           mode,
                           beam_size,
                           batch_idx):
        '''
        Produces the code comment given a code snippet step by step and evaluates
        the generated comment against the reference summary.

        Args:
            src: code snippet numericalized
            target_vocab: The vocabulary built from the summaries in training set.
            token: The token adjacency matrices. 
                   Shape: `(batch_size, max_src_len, max_src_len)`
            statement: The statement adjacency matrices. 
                       Shape: `(batch_size, max_src_len, max_src_len)`
            data_flow: The data flow adjacency matrices. 
                       Shape: `(batch_size, max_src_len, max_src_len)`
            control_flow: The control flow adjacency matrices. 
                          Shape: `(batch_size, max_src_len, max_src_len)`
            ast: The ast adjacency matrices. 
                 Shape: `(batch_size, max_src_len, max_src_len)`
            code (string): code snippet in textual form.
            summary (string): reference code comment.
            log: file to write the evaluation metrics of the generated summaries.
            scorer: A RougeScorer object to compute ROUGE-L.
            mode (string): Indicates whether we only want to compute validation loss or if we also
                           want to translate the source sentences in the validation set
                           (either using greedy decoding or beam search).
                           Can be one of the following: "loss", "greedy" or "beam"
            beam_size (int): Number of elements to store during beam search
                             Only applicable if `mode == 'beam'`
            batch_idx (int): The batch number, indicating how many batches were
                             already processed before. Used to give a unique ID
                             for each predicted and reference summary. 
        Returns:
            The average BLEU, METEOR, ROUGE-L F-measure, Precision, Recall and 
            F1-score of the predicted sentence relative to the reference. 

            It also returns all predicted summaries and respective references in two 
            different dictionaries. These dictionaries will be used to compute the 
            BLEU, METEOR and ROUGE-L according to the libraries used by NeuralCodeSum
            and related works.
        '''
        start_symbol_idx = target_vocab.token_to_idx['<BOS>']
        end_symbol_idx = target_vocab.token_to_idx['<EOS>']

        sum_bleu, sum_meteor, sum_rouge_l, sum_precision, sum_recall, sum_f1 = 0, 0, 0, 0, 0, 0

        batch_size = src.shape[0]

        hyps = {}
        refs = {}

        if mode == 'greedy':
            tgt_preds_idx = greedy_decode(self.model,
                                          src,
                                          source_ids,
                                          source_mask,
                                          token,
                                          statement,
                                          data_flow,
                                          control_flow,
                                          ast,
                                          self.device,
                                          start_symbol_idx,
                                          end_symbol_idx,
                                          self.max_tgt_length)

        # Translating a batch of source sentences
        for i in range(batch_size):
            if mode == 'greedy':
                # Translating the indexes of tokens to the textual
                # representation of tokens. Replacing <BOS> and <EOS>
                # with empty string
                tgt_pred_tokens = translate_tokens(tgt_preds_idx[i], target_vocab)
            elif mode == 'beam':
                # src[i] has shape (max_src_length, )
                # Performing an unsqueeze on src[i] will make its shape (1, max_src_length)
                # which is the correct shape since batch_size = 1 in this case
                # The same happens with token, statement, data flow, control flow and ast
                tgt_preds_idx = beam_search(self.model,
                                            src[i].unsqueeze(0),
                                            self.device,
                                            start_symbol_idx,
                                            end_symbol_idx,
                                            self.max_tgt_length,
                                            beam_size)
                
                tgt_preds_idx = tgt_preds_idx.flatten()
                tgt_pred_tokens = translate_tokens(tgt_preds_idx, target_vocab)
            else:
                raise ValueError("The mode " + mode + " does not exist.")

            # Getting tokens of the reference and prediction to
            # compute METEOR
            reference = summary[i].split()
            prediction = tgt_pred_tokens.split()

            # Creating sets to compute traditional metrics:
            # precision, recall and F1 score (required by NLTK package)
            reference_set = set(reference)
            prediction_set = set(prediction)

            example = {}
            example["prediction"] = tgt_pred_tokens
            example["reference"] = summary[i]
            example["code"] = code[i]

            hyps[batch_idx*batch_size + i] = [tgt_pred_tokens]
            refs[batch_idx*batch_size + i] = [summary[i]]

            precision_score = precision(reference_set, prediction_set)
            if precision_score is None:
                precision_score = 0.0

            f_score = f_measure(reference_set, prediction_set)
            if f_score is None:
                f_score = 0.0

            method2 = SmoothingFunction().method2
            # Sentence BLEU requires a list(list(str)) for the references
            example["BLEU"] = sentence_bleu(
                [reference], prediction, smoothing_function=method2)
            example["METEOR"] = single_meteor_score(reference, prediction)

            # Getting the ROUGE-L F1-score (and ignoring the ROUGE-L Precision/Recall
            # scores)
            example["ROUGE-L"] = scorer.score(summary[i],
                                              tgt_pred_tokens)['rougeL'].fmeasure
            example["Precision"] = precision_score
            example["Recall"] = recall(reference_set, prediction_set)
            example["F1"] = f_score

            log.write(json.dumps(example, indent=4) + ',\n')

            sum_bleu += example["BLEU"]
            sum_meteor += example["METEOR"]
            sum_rouge_l += example["ROUGE-L"]
            sum_precision += example["Precision"]
            sum_recall += example["Recall"]
            sum_f1 += example["F1"]

        return sum_bleu / batch_size, sum_meteor / batch_size, sum_rouge_l / batch_size, \
            sum_precision / batch_size, sum_recall / batch_size, sum_f1 / batch_size, hyps, refs

    def get_model(self):
        '''
        Function that obtains the module of the model when we're running it in
        parallel with one or more GPUs.
        It's only used in load/save operations.
        '''
        if self.gpu_rank is not None:
            return self.model.module
        else:
            return self.model

    def get_map_location(self, gpu_rank):
        '''
        Specifies how to map the storage location for devices 
        (used to load the model). It is useful when we are running the
        program with multiple GPUs, to tell to the GPUs 1,2,... that the
        model is stored in a location mapped by GPU 0 (because that was the
        GPU that stored the model).

        Args:
            gpu_rank (int): The rank of the GPU.
                            It has the value of None if no GPUs are avaiable or
                            only 1 GPU is available.

        Returns:
            A dictionary with a mapping between devices and storage
        '''
        if self.gpu_rank is not None:
            return {'cuda:%d' % 0: 'cuda:%d' % gpu_rank}
        else:
            return None

    def save(self, last_epoch=False):
        '''
        Stores the model learned parameters (weights) in a file.
        '''
        model = self.get_model()
        try:
            if last_epoch:
                filename = '../results/model_weights_last.pth'
            else:
                filename = '../results/model_weights.pth'
            torch.save(model.state_dict(), filename)
        except Exception:
            traceback.print_exc()
            logger.error("It was not possible to save the model weights.")
            sys.exit(1)

    def save_checkpoint(self, epoch, training_losses, validation_losses, best_bleu):
        '''
        Stores a checkpoint of the model, the optimizer state, current epoch and
        training and validation losses from first epoch to current one.

        Args:
            epoch (int): Current epoch
            training_losses: List of training losses
            validation_losses: List of validation losses
            best_bleu (int): The best BLEU metric obtained so far in validation set
        '''
        model = self.get_model()
        try:
            params = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_losses': training_losses,
                'validation_losses': validation_losses,
                'best_bleu': best_bleu
            }
            torch.save(params, '../results/model_weights_checkpoint.pth')
        except Exception:
            traceback.print_exc()
            logger.error("It was not possible to save a model checkpoint.")
            sys.exit(1)

    def load(self, gpu_rank):
        '''
        Loads the model learned parameters from a saved file.
        '''
        self.model = self.get_model()
        map_location = self.get_map_location(gpu_rank)
        try:
            self.model.load_state_dict(torch.load('../results/model_weights_last.pth',
                                                  map_location=map_location))
        except Exception:
            traceback.print_exc()
            logger.error("It was not possible to load the model weights from the " +
                         "specified filename.")
            sys.exit(1)

    def load_checkpoint(self, gpu_rank):
        '''
        Loads the last model checkpoint.

        Returns:
            epoch (int): Epoch of the checkpoint
            training_losses: List with the losses during training from first epoch
                             to epoch of the checkpoint
            validation_losses: List with the losses during validation from first epoch
                               to epoch of the checkpoint
            best_bleu (int): The best BLEU metric obtained in validation set
        '''
        self.model = self.get_model()
        map_location = self.get_map_location(gpu_rank)
        try:
            checkpoint = torch.load('../results/model_weights_checkpoint.pth',
                                    map_location=map_location)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            training_losses = checkpoint['training_losses']
            validation_losses = checkpoint['validation_losses']
            epoch = checkpoint['epoch']
            best_bleu = checkpoint['best_bleu']
            return epoch, training_losses, validation_losses, best_bleu
        except Exception:
            traceback.print_exc()
            logger.error("It was not possible to load the model from the " +
                         "checkpoint.")
            sys.exit(1)

from datetime import datetime
import json
import logging
import sys
import traceback
from nltk.metrics.scores import precision, recall, f_measure
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
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
                 gpu_rank):
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
                                 device)

        # Passing the model and all its layers to GPU if available
        self.model = self.model.to(device)

        if gpu_rank is not None:
            # Used to run model in several GPUs
            self.model = DDP(self.model,
                             device_ids=[gpu_rank],
                             output_device=gpu_rank)

        # Optimizes the model. TODO: Try to fix this!
        # self.model = torch.compile(self.model)

        # Function used to compute the loss. ignore_index is the padding token index
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx,
                                             label_smoothing=label_smoothing)

        # TODO: Test with SGD optimizer
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=learning_rate,
                                    betas=(0.9, 0.98),
                                    eps=1e-9)

        # Adjusts the learning rate during training
        # Implements the "Noam" learning rate scheduler, used in vanilla Transformer
        self.scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda step: Model.rate(
                step, model_size=d_model, factor=1.0, warmup=400
            ),
        )

        self.device = device
        self.gpu_rank = gpu_rank

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

    def train_epoch(self, train_dataloader, tgt_vocab_size, grad_clipping):
        '''
        Trains the model during one epoch, using the examples of the dataset for training.
        Computes the loss during this epoch.

        Args:
            train_dataloader: A Dataloader object that contains the training set.
            tgt_vocab_size (int): size of the target vocabulary.
            grad_clipping (int): Maximum norm of the gradient.

        Returns:
            The average loss across all examples during one epoch.

        Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
                https://medium.com/@hunter-j-phillips/putting-it-all-together-the-implemented-transformer-bfb11ac1ddfe
        '''
        # Set the model to training mode
        self.model.train()
        losses = 0

        for src, tgt in tqdm(train_dataloader, desc="Training"):
            # Passing vectors to GPU if it's available
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Zeroing the gradients of transformer parameters
            self.optimizer.zero_grad()

            # Slicing the last element of tgt_data because it is used to compute the
            # loss.
            output = self.model(src, tgt[:, :-1])

            # - output is reshaped using view() method in shape (batch_size * tgt_len, tgt_vocab_size)
            # - tgt_data is reshaped using view() method in shape (batch_size * tgt_len)
            # - criterion(prediction_labels, target_labels)
            # - tgt[:, 1:] removes the first token from the target sequence since we are training the
            # model to predict the next word based on previous words.
            loss = self.criterion(output.contiguous().view(-1, tgt_vocab_size),
                                  tgt[:, 1:].contiguous().view(-1))

            # Computing gradients of loss through backpropagation
            loss.backward()

            # clip the weights
            clip_grad_norm_(self.model.parameters(), grad_clipping)

            # Update the model's parameters based on the computed gradients
            self.optimizer.step()

            # Adjust learning rate
            self.scheduler.step()

            losses += loss.item()

        return losses / len(list(train_dataloader))

    def validate(self,
                 val_dataloader,
                 mode,
                 target_vocab,
                 tgt_vocab_size,
                 max_tgt_length,
                 num_epoch,
                 beam_size):
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
            tgt_vocab_size (int): size of the target vocabulary.
            max_tgt_length (int): The maximum length of the summaries.
            num_epoch (int): The current training epoch.
            beam_size (int): Number of elements to store during beam search
                             Only applicable if `mode == 'beam'`

        Returns:
            The average loss across all examples during one epoch.

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
                       '_gpu' + str(self.gpu_rank) + '_epoch' + str(num_epoch) + '.json', 'w')

        # evaluate without updating gradients
        # Tells pytorch to not calculate the gradients
        with torch.no_grad():
            for code, summary, src, tgt in tqdm(val_dataloader, desc="Validating"):
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                if mode in ['beam', 'greedy']:
                    self.translate_sentence(src,
                                            target_vocab,
                                            max_tgt_length,
                                            code,
                                            summary,
                                            log,
                                            scorer,
                                            mode,
                                            beam_size)

                # Slicing the last element of tgt_data because it is used to compute the
                # loss.
                output = self.model(src, tgt[:, :-1])

                # - output is reshaped using view() method in shape (batch_size * tgt_len, tgt_vocab_size)
                # - tgt_data is reshaped using view() method in shape (batch_size * tgt_len)
                # - criterion(prediction_labels, target_labels)
                # - tgt[:, 1:] removes the first token from the target sequence since we are training the
                # model to predict the next word based on previous words.
                loss = self.criterion(output.contiguous().view(-1, tgt_vocab_size),
                                      tgt[:, 1:].contiguous().view(-1))

                losses += loss.item()

        if mode in ['beam', 'greedy']:
            log.close()

        return losses / len(list(val_dataloader))

    def test(self,
             test_dataloader,
             target_vocab,
             max_tgt_length,
             mode,
             beam_size):
        '''
        Tests the model with the testing set.

        Args:
            test_dataloader: A Dataloader object that contains the testing set.
            target_vocab: The vocabulary built from the summaries in training set.
            max_tgt_length (int): The maximum length of the summaries.
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

        number_examples = 0
        sum_bleu, sum_meteor, sum_rouge_l, sum_precision, sum_recall, sum_f1 = 0, 0, 0, 0, 0, 0

        # evaluate without updating gradients
        # Tells pytorch to not calculate the gradients
        with torch.no_grad():
            with open('../results/test_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") +
                      '_' + str(self.gpu_rank) + '.json', 'w') as log:
                for code, summary, src, tgt in tqdm(test_dataloader, desc="Testing"):
                    src = src.to(self.device)
                    tgt = tgt.to(self.device)

                    batch_size = src.shape[0]

                    bleu, meteor, rouge_l, \
                        precision, recall, f1 = self.translate_sentence(src,
                                                                        target_vocab,
                                                                        max_tgt_length,
                                                                        code,
                                                                        summary,
                                                                        log,
                                                                        scorer,
                                                                        mode,
                                                                        beam_size)

                    # Performing weighted average of metrics
                    sum_bleu += bleu * batch_size
                    sum_meteor += meteor * batch_size
                    sum_rouge_l += rouge_l * batch_size
                    sum_precision += precision * batch_size
                    sum_recall += recall * batch_size
                    sum_f1 += f1 * batch_size

                    number_examples += batch_size

        logger.info(f"Metrics:\nBLEU: {sum_bleu / number_examples:7.3f} | " +
                    f"METEOR: {sum_meteor / number_examples:7.3f} | " +
                    f"ROUGE-L: {sum_rouge_l / number_examples:7.3f} | " +
                    f"Precision: {sum_precision / number_examples:7.3f} | " +
                    f"Recall: {sum_recall / number_examples:7.3f} | " +
                    f"F1: {sum_f1 / number_examples:7.3f}")

    def translate_sentence(self,
                           src,
                           target_vocab,
                           max_tgt_length,
                           code,
                           summary,
                           log,
                           scorer,
                           mode,
                           beam_size):
        '''
        Produces the code comment given a code snippet step by step and evaluates
        the generated comment against the reference summary.

        Args:
            src: code snippet numericalized
            target_vocab: The vocabulary built from the summaries in training set.
            max_tgt_length (int): Maximum length of the generated summary.
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

        Returns:
            The average BLEU, METEOR, ROUGE-L F-measure, Precision, Recall and 
            F1-score of the predicted sentence relative to the reference. 
        '''
        start_symbol_idx = target_vocab.token_to_idx['<BOS>']
        end_symbol_idx = target_vocab.token_to_idx['<EOS>']

        sum_bleu, sum_meteor, sum_rouge_l, sum_precision, sum_recall, sum_f1 = 0, 0, 0, 0, 0, 0

        batch_size = src.shape[0]

        # Translating a batch of source sentences
        for i in range(batch_size):
            # src[i] has shape (max_src_length, )
            # Performing an unsqueeze on src[i] will make its shape (1, max_src_length)
            # which is the correct shape since batch_size = 1 in this case
            if mode == 'greedy':
                tgt_preds_idx = greedy_decode(self.model,
                                              src[i].unsqueeze(0),
                                              self.device,
                                              start_symbol_idx,
                                              end_symbol_idx,
                                              max_tgt_length)
            elif mode == 'beam':
                tgt_preds_idx = beam_search(self.model,
                                            src[i].unsqueeze(0),
                                            self.device,
                                            start_symbol_idx,
                                            end_symbol_idx,
                                            max_tgt_length,
                                            beam_size)
            else:
                raise ValueError("The mode " + mode + " does not exist.")

            # Passing the tensor to 1 dimension
            tgt_preds_idx.flatten()

            # Translating the indexes of tokens to the textual
            # representation of tokens Replacing <BOS> and <EOS>
            # with empty string
            tgt_pred_tokens = translate_tokens(tgt_preds_idx, target_vocab)

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

            precision_score = precision(reference_set, prediction_set)
            if precision_score is None:
                precision_score = 0.0

            f_score = f_measure(reference_set, prediction_set)
            if f_score is None:
                f_score = 0.0

            # Sentence BLEU requires a list(list(str)) for the references
            example["BLEU"] = sentence_bleu([reference], prediction)
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
            sum_precision / batch_size, sum_recall / batch_size, sum_f1 / batch_size

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

    def save(self):
        '''
        Stores the model learned parameters (weights) in a file.
        '''
        model = self.get_model()
        try:
            torch.save(model.state_dict(), '../results/model_weights.pth')
        except Exception:
            traceback.print_exc()
            logger.error("It was not possible to save the model weights.")
            sys.exit(1)

    def save_checkpoint(self, epoch, training_losses, validation_losses):
        '''
        Stores a checkpoint of the model, the optimizer state, current epoch and
        training and validation losses from first epoch to current one.

        Args:
            epoch (int): Current epoch
            training_losses: List of training losses
            validation_losses: List of validation losses
        '''
        model = self.get_model()
        try:
            params = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_losses': training_losses,
                'validation_losses': validation_losses
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
            self.model.load_state_dict(torch.load('../results/model_weights.pth',
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
            return epoch, training_losses, validation_losses
        except Exception:
            traceback.print_exc()
            logger.error("It was not possible to load the model from the " +
                  "checkpoint.")
            sys.exit(1)

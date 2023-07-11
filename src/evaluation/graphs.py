from datetime import datetime
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Multiple calls to getLogger() with the same name will return a reference
# to the same Logger object, which saves us from passing the logger objects
# to every part where it’s needed.
# Source: https://realpython.com/python-logging/
logger = logging.getLogger('main_logger')
# To avoid having repeated logs!
logger.propagate = False


def display_attention(input,
                      output,
                      attention,
                      label,
                      n_heads):
    '''
    Display the attention matrix for each head of a sequence.

    Args:
        input: The input given to the model (e.g. code snippet).
        output: The reference/predicted output (e.g. short code comment).
        attention: attention scores for the heads. Shape: `(batch_size, num_heads, query_len, key_len)`
        label (string): part of the filename of the attention matrices.
                        Used to identifier whether the matrices are from encoder self-attn, decoder self-attn
                        or decoder cross-attn.
        n_heads (int): number of heads.

    Source: https://medium.com/@hunter-j-phillips/putting-it-all-together-the-implemented-transformer-bfb11ac1ddfe
            https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
            https://matplotlib.org/stable/gallery/misc/multipage_pdf.html
    '''
    logger.info(f"Creating an attention heatmap for {label}")

    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages('../results/' + label + "_" +
                  datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.pdf') as pdf:

        for i in range(n_heads):
            # select the respective head and make it a numpy array for plotting
            _attention = attention.squeeze(0)[i, :len(output), :len(input)] \
                                  .cpu().detach().numpy()

            # create a plot
            # width and height depend on the number of squares in the heatmap
            # respective axis
            height = 0.6 * _attention.shape[0]
            width = 0.6 * _attention.shape[1]
            _, ax = plt.subplots(figsize=(width, height))

            # plot the matrix. cmap defines the colors of the matrix
            cax = ax.matshow(_attention, cmap='viridis', aspect='auto')

            # create side bar with attention score
            cbar = ax.figure.colorbar(cax, ax=ax)
            cbar.ax.set_ylabel("Attention score", rotation=-90, va="bottom")

            # # set the size of the labels
            # ax.tick_params(labelsize=6)

            # set the indices for the tick marks
            ax.set_xticks(range(len(input)))
            ax.set_yticks(range(len(output)))

            # if the provided sequences are sentences or indices
            if isinstance(input[0], str):
                ax.set_xticklabels([t for t in input], rotation=45)
                ax.set_yticklabels(output)
            elif isinstance(input[0], int):
                ax.set_xticklabels(input, rotation=45)
                ax.set_yticklabels(output)
            
            # Loop over data dimensions and create text annotations.
            for i in range(len(output)):
                for j in range(len(input)):
                    ax.text(j, i, "{:.2f}".format(_attention[i, j]), ha="center", va="center", color="w")


            plt.title('Head ' + str(i))
            plt.tight_layout()
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()


def create_loss_plot(train_epoch_loss, val_epoch_loss, gpu_rank):
    '''
    Plots the training and validation loss.

    Args:
        train_epoch_loss (list): The training loss for each epoch.
                                 Size: number of epochs.
        val_epoch_loss (list): The validation loss for each epoch.
                               Size: number of epochs.
        gpu_rank (int): The rank of the GPU.
                        It has the value of None if no GPUs are avaiable or
                        only 1 GPU is available.

    Source: https://machinelearningmastery.com/plotting-the-training-and-validation-loss-curves-for-the-transformer-model/
    '''
    epochs = range(1, len(train_epoch_loss) + 1)

    # Plot and label the training loss values
    plt.plot(epochs, train_epoch_loss, label='Training Loss')
    plt.plot(epochs, val_epoch_loss, label='Validation Loss')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Optional - How to set the tick locations
    # step = 10
    # plt.xticks(arange(0, len(epoch_loss) + step, step))

    # Add the legend of line in plot
    plt.legend(loc='best')

    plt.savefig("../results/" + "train_val_loss" + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") +
                '_' + str(gpu_rank) + '.png')

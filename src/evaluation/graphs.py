from datetime import datetime
import matplotlib.pyplot as plt


def display_attention(input, output, attention, label, n_heads=8, n_rows=4, n_cols=2):
    '''
    Display the attention matrix for each head of a sequence.

    Args:
        input: The input given to the model (e.g. code snippet).
        output: The output predicted by the model (e.g. short code comment).
        attention: attention scores for the heads. Shape: `(batch_size, num_heads, query_len, key_len)`
        label (string): part of the filename of the attention matrices.
                        Used to identifier whether the matrices are from encoder self-attn, decoder self-attn
                        or decoder cross-attn.
        n_heads (int): number of heads.
        n_rows (int): number of rows. Default is 8, meaning that we will have 8 rows, each one with the
                      matrices.
        n_cols (int): number of columns. Default is 1, meaning that we will have only 1 matrix per row.

    Source: https://medium.com/@hunter-j-phillips/putting-it-all-together-the-implemented-transformer-bfb11ac1ddfe
            https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    '''
    # ensure the number of rows and columns are equal to the number of heads
    assert n_rows * n_cols == n_heads

    # figure size
    fig = plt.figure(figsize=(15, 25))

    # visualize each head
    for i in range(n_heads):

        # create a plot
        ax = fig.add_subplot(n_rows, n_cols, i+1)

        # select the respective head and make it a numpy array for plotting
        _attention = attention.squeeze(0)[i, :, :].cpu().detach().numpy()

        # plot the matrix. cmap defines the colors of the matrix
        cax = ax.matshow(_attention, cmap='viridis', aspect='auto')

        # create side bar with attention score
        cbar = ax.figure.colorbar(cax, ax=ax)
        cbar.ax.set_ylabel("Attention score", rotation=-90, va="bottom")

        # set the size of the labels
        ax.tick_params(labelsize=12)

        # set the indices for the tick marks
        ax.set_xticks(range(len(input)))
        ax.set_yticks(range(len(output)))

        # if the provided sequences are sentences or indices
        if isinstance(input[0], str):
            ax.set_xticklabels([t.lower() for t in input], rotation=45)
            ax.set_yticklabels(output)
        elif isinstance(input[0], int):
            ax.set_xticklabels(input, rotation=45)
            ax.set_yticklabels(output)

    plt.savefig('../results/' + label + "_" +
                datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.png')


def create_loss_plot(train_epoch_loss, val_epoch_loss, gpu_rank):
    '''
    Plots the training and validation loss.

    Args:
        train_epoch_loss (list): The training loss for each epoch.
                                 Size: number of epochs.
        val_epoch_loss (list): The validation loss for each epoch.
                               Size: number of epochs.
        gpu_rank (int): The rank of the GPU.
                        It has the value of -1 if no GPUs are avaiable.

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

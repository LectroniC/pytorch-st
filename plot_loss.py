import numpy as np
import sys
from matplotlib import pyplot as plt 
from matplotlib.pyplot import figure

def parse_log_file(file_name):
    """
    log_line = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        time.ctime(), epoch_num + 1, sample_count, dataset_length, batch_num+1,
                        cumulate_content_loss /
                        (batch_num+1.0), cumulate_style_loss /
                        (batch_num+1.0), cumulate_tv_loss/(batch_num+1.0),
                        content_loss, style_loss, tv_loss
                    )
    """
    losses = []
    # Using readlines()
    file1 = open(file_name, 'r')
    lines = file1.readlines()
    for l in lines:
        # Remove new line
        l = l[:-1]
        tokens = l.split(",")
        content_loss = tokens[-3]
        style_loss = tokens[-2]
        tv_loss = tokens[-1]
        total_loss = content_loss + style_loss + tv_loss
        losses.append(total_loss)

def draw_loss_plot_simple(losses, fig_file_name="loss_graph.pdf",granularity=10):
    figure(figsize=(9, 5))
    losses = losses[::granularity]
    x = np.arange(1,len(losses)+1) 
    y = np.asarray(losses)
    plt.title("Loss curve") 
    plt.xlabel("Samples") 
    plt.ylabel("Total Losses") 
    plt.plot(x,y)
    plt.grid()
    plt.savefig(fig_file_name)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: Invalid Params")
        exit()
    log_file_name = sys.argv[1]
    losses = parse_log_file(log_file_name)
    draw_loss_plot_simple(losses)
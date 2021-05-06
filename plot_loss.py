import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from torch.nn.modules import loss


def parse_log_file(file_name, take_cumulate=False, losses_type="total"):
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

        if take_cumulate == False:
            content_loss = float(tokens[-3])
            style_loss = float(tokens[-2])
            tv_loss = float(tokens[-1])
        else:
            content_loss = float(tokens[-6])
            style_loss = float(tokens[-5])
            tv_loss = float(tokens[-4])

        total_loss = content_loss + style_loss + tv_loss
        if losses_type=="total":
            losses.append(total_loss)
        if losses_type=="content":
            losses.append(content_loss)
        if losses_type=="style":
            losses.append(style_loss)
        if losses_type=="tv":
            losses.append(tv_loss)
    return losses


def draw_loss_plot_simple(input_log_path="./loss/plst_la_muse.csv", fig_file_name="loss_graph.pdf", granularity=10):
    losses = parse_log_file(input_log_path)
    figure(figsize=(9, 5))
    losses = losses[::granularity]
    x = np.arange(1, len(losses)+1)
    y = np.asarray(losses)
    plt.title("Loss curve")
    plt.xlabel("Samples")
    plt.ylabel("Total Losses")
    plt.plot(x, y)
    plt.grid()
    plt.savefig(fig_file_name)


def draw_loss_plot_multiple(input_log_paths=["./loss/msgnet_256.csv", "./loss/plst_la_muse.csv", "./loss/plst_starry_night.csv"], 
                            fig_file_name="loss_graph_all.pdf", 
                            tick_granularity=10, 
                            smooth_level=10, 
                            take_cumulate=False,
                            y_label_title="Total Losses",
                            losses_type="total"):
    figure(figsize=(9, 5))
    for log_path in input_log_paths:
        losses = parse_log_file(log_path, take_cumulate=take_cumulate, losses_type=losses_type)
        losses = [sum(losses[i:min(i+smooth_level,len(losses))])*1.0 /
                  min(smooth_level,len(losses)-i) for i in range(0, len(losses), smooth_level)]
        y = np.asarray(losses)
        plt.title("Loss Curve")
        plt.xlabel("Samples (Each taken per 100 batches) Averaged every {} samples".format(smooth_level))
        plt.ylabel(y_label_title)
        plt.plot(y, label=log_path)

    plt.xticks(np.arange(0, len(losses), tick_granularity))
    plt.xlim(xmin=0)
    plt.legend()
    plt.grid()
    plt.savefig(fig_file_name)


if __name__ == '__main__':
    draw_loss_plot_multiple(fig_file_name="loss_graph_all.pdf",y_label_title="Total Losses", losses_type="total")
    draw_loss_plot_multiple(fig_file_name="loss_graph_content.pdf",y_label_title="Content Losses", losses_type="total")
    draw_loss_plot_multiple(fig_file_name="loss_graph_style.pdf",y_label_title="Style Losses", losses_type="content")
    draw_loss_plot_multiple(fig_file_name="loss_graph_tv.pdf",y_label_title="TV Losses", losses_type="tv")

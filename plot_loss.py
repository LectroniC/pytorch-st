import numpy as np
import sys
from matplotlib import pyplot as plt 
from matplotlib.pyplot import figure

def parse_log_file(file_name, take_cumulate=False):
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
        
        if take_cumulate==False:
            content_loss = float(tokens[-3])
            style_loss = float(tokens[-2])
            tv_loss = float(tokens[-1])
        else:
            content_loss = float(tokens[-6])
            style_loss = float(tokens[-5])
            tv_loss = float(tokens[-4])

        total_loss = content_loss + style_loss + tv_loss
        if total_loss>50 and "msgnet_all" in file_name:
            print(l)
        losses.append(total_loss)
    return losses

def draw_loss_plot_simple(input_log_path="./loss/plst_la_muse.csv",fig_file_name="loss_graph.pdf",granularity=10):
    losses = parse_log_file(input_log_path)
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

def draw_loss_plot_multiple(input_log_paths=["./loss/msgnet_all.csv", "./loss/plst_la_muse.csv", "./loss/plst_starry_night.csv"], fig_file_name="loss_graph_all.pdf",tick_granularity=10,smooth_level=10,take_cumulate=False):
    figure(figsize=(9, 5))
    for log_path in input_log_paths:
        losses = parse_log_file(log_path,take_cumulate=take_cumulate)
        losses = [sum(losses[i:i+smooth_level])*1.0/smooth_level for i in range(0,len(losses),smooth_level)]
        y = np.asarray(losses)
        plt.title("Loss Curve")
        plt.xlabel("Samples (Each taken per 100 batches)") 
        plt.ylabel("Total Losses") 
        plt.plot(y, label=log_path)
    
    plt.xticks(np.arange(0, len(losses), tick_granularity))
    plt.xlim(xmin=0)
    plt.legend()
    plt.grid()
    plt.savefig(fig_file_name)

if __name__ == '__main__':
    #draw_loss_plot_multiple(smooth_level=1,take_cumulate=True)
    draw_loss_plot_multiple()
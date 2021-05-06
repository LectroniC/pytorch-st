import numpy as np
import torch
import os
import argparse
import time

from PIL import Image
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from dataset.loader import get_simple_dataset_transform
from models.plst import StyleTransferNet, Vgg16, Loss_plst
from models.msgnet import MSGNet, Loss_msg
from torch.nn.functional import interpolate

# Reference: https://github.com/dxyang/StyleTransfer/blob/master/style.py
# Global Variables
BATCH_SIZE = 4
LEARNING_RATE = 1e-3    
EPOCHS = 3

REPORT_BATCH_FREQ = 1000
LOG_LOSS_VALUE_FREQ = 100
CHECKPOINT_SAVE_EPOCH_FREQ = 1

# Opens and returns image file as a PIL image (0-255)

def load_image(filename):
    img = Image.open(filename)
    return img

# The image tensor has dimension (c, h, w)
# Assume the output was produced by normalized data.


def restore_and_save_image(filename, data):
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.clone().numpy()
    img = ((img * std + mean).transpose(1, 2, 0)
           * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def save_model(model, use_gpu, model_name):
    # save model
    model.eval()

    if use_gpu:
        model.cpu()

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    filename = "saved_models/" + str(model_name) + ".state"
    torch.save(model.state_dict(), filename)

    if use_gpu:
        model.cuda()

    model.train()


def train(args):
    # GPU enabling
    if (args.gpu != None):
        use_gpu = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current CUDA device: {}".format(torch.cuda.current_device()))
    else:
        dtype = torch.FloatTensor
        use_gpu = False

    # visualization of training controlled by flag
    if (args.visualization_freq != 0):
            simple_transform = get_simple_dataset_transform(256)

            img_avocado = load_image("sample_images/avocado.jpg")
            img_avocado = simple_transform(img_avocado)
            img_avocado = Variable(img_avocado.repeat(
                1, 1, 1, 1), requires_grad=False).type(dtype)

            img_cheetah = load_image("sample_images/cheetah.jpg")
            img_cheetah = simple_transform(img_cheetah)
            img_cheetah = Variable(img_cheetah.repeat(
                1, 1, 1, 1), requires_grad=False).type(dtype)

            img_quad = load_image("sample_images/quad.jpg")
            img_quad = simple_transform(img_quad)
            img_quad = Variable(img_quad.repeat(1, 1, 1, 1),
                                requires_grad=False).type(dtype)

    if args.model_name == "plst":

        simple_transform = get_simple_dataset_transform(256)
        print("Training model PLST...")
        
        print("Dataset folder "+args.dataset)
        train_dataset = datasets.ImageFolder(args.dataset, simple_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        dataset_length = len(train_dataset)
        print("Loaded total images: "+str(dataset_length))

        # load style image
        print("Style Image "+args.style_image)
        style = load_image(args.style_image)
        style = simple_transform(style)
        style = Variable(style.repeat(BATCH_SIZE, 1, 1, 1)).type(dtype)

        # define network
        image_transformer = StyleTransferNet().type(dtype)
        optimizer = Adam(image_transformer.parameters(), LEARNING_RATE)

        # Initialize vgg network for loss
        vgg = Vgg16().type(dtype)

        loss_plst = Loss_plst(vgg, style, \
            lambda_c=args.c, 
            lambda_s=args.s,  
            lambda_tv=args.tv
        )

        best_total_loss = None
        total_batch_num = 0
        sample_count = 0
        cumulate_content_loss = 0
        cumulate_style_loss = 0
        cumulate_tv_loss = 0

        for epoch_num in range(EPOCHS):
            # train network
            image_transformer.train()
            for _, (x, label) in enumerate(train_loader):

                # Forward
                optimizer.zero_grad()
                x = Variable(x).type(dtype)
                y_hat = image_transformer(x)
                content_loss, style_loss, tv_loss = loss_plst.extract_and_calculate_loss(x, y_hat)

                cumulate_content_loss += content_loss
                cumulate_style_loss += style_loss
                cumulate_tv_loss += tv_loss
                total_loss = content_loss + style_loss + tv_loss

                # Backprop
                total_loss.backward()
                optimizer.step()

                sample_count += len(x)
                total_batch_num += 1

                # Showing training message (Could incorporate other backends in the future)
                if ((total_batch_num + 1) % REPORT_BATCH_FREQ == 0):
                    status = "Time: {}\n  Epoch {}:  [{}/{}]  Batch:[{}]  AvgContentLoss: {:.5f}  AvgStyleLoss: {:.5f}  AvgTVLoss: {:.5f}  content: {:.5f}  style: {:.5f}  tv: {:.5f} \n".format(
                        time.ctime(), epoch_num + 1, sample_count, dataset_length, total_batch_num+1,
                        cumulate_content_loss /
                        (total_batch_num+1.0), cumulate_style_loss /
                        (total_batch_num+1.0), cumulate_tv_loss/(total_batch_num+1.0),
                        content_loss, style_loss, tv_loss
                    )
                    print(status)
                
                if args.loss_log_path is not None:
                    log_line = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        time.ctime(), epoch_num + 1, sample_count, dataset_length, total_batch_num+1,
                        cumulate_content_loss /
                        (total_batch_num+1.0), cumulate_style_loss /
                        (total_batch_num+1.0), cumulate_tv_loss/(total_batch_num+1.0),
                        content_loss, style_loss, tv_loss
                    )
                    if((total_batch_num + 1) % LOG_LOSS_VALUE_FREQ==0):
                        with open(args.loss_log_path, "a") as f:
                            f.write(log_line)

                if args.visualization_freq != 0 and ((total_batch_num + 1) % args.visualization_freq == 0):
                    print("Write vis images to folder.")

                    image_transformer.eval()

                    folder_name = args.model_name+"_"+args.visualization_folder_id
                    if not os.path.exists("visualization"):
                        os.makedirs("visualization")
                    if not os.path.exists("visualization/{}".format(folder_name)):
                        os.makedirs("visualization/{}".format(folder_name))

                    output_img_1 = image_transformer(img_avocado).cpu()
                    output_img_1_path = (
                        "visualization/{}/{}_{}_img_avocado.jpg".format(folder_name, str(epoch_num+1), str(total_batch_num+1)))
                    restore_and_save_image(
                        output_img_1_path, output_img_1.data[0])

                    output_img_2 = image_transformer(img_cheetah).cpu()
                    output_img_2_path = "visualization/{}/{}_{}_img_cheetah.jpg".format(
                        folder_name, str(epoch_num+1), str(total_batch_num+1))
                    restore_and_save_image(
                        output_img_2_path, output_img_2.data[0])

                    output_img_3 = image_transformer(img_quad).cpu()
                    output_img_3_path = "visualization/{}/{}_{}_img_quad.jpg".format(
                        folder_name, str(epoch_num+1), str(total_batch_num+1))
                    restore_and_save_image(
                        output_img_3_path, output_img_3.data[0])

                    image_transformer.train()
            # Save model
            if ((epoch_num + 1) % CHECKPOINT_SAVE_EPOCH_FREQ == 0):
                save_model(image_transformer, use_gpu,
                           args.model_name+"_"+args.model_id+"_"+str(epoch_num + 1))

        # save model
        save_model(image_transformer, use_gpu, args.model_name+"_"+args.model_id+"_final")

    if args.model_name == "msgnet":
        # MSG Net training pipeline
        print("Training model MSGNet...")
        # visualization of training controlled by flag

        # content image dataset loader
        print("Content dataset folder "+args.dataset)
        content_transform = get_simple_dataset_transform(256)
        train_dataset = datasets.ImageFolder(args.dataset, content_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        dataset_length = len(train_dataset)
        print("Loaded total images: "+str(dataset_length))

        # style image dataset loader
        print("Style dataset folder"+args.style_image)
        style_transform = get_simple_dataset_transform(256)
        style_dataset = datasets.ImageFolder(args.style_image, style_transform)
        style_loader = DataLoader(style_dataset, batch_size=1, shuffle=True)
        style_length = len(style_dataset)
        print("Loaded style images: "+str(style_length))

        # define network
        image_transformer = MSGNet(block_size=128).type(dtype)
        optimizer = Adam(image_transformer.parameters(), LEARNING_RATE)

        # Initialize vgg network for loss
        vgg = Vgg16().type(dtype)
        loss_msg = Loss_msg(vgg, \
            lambda_c=args.c, 
            lambda_s=args.s,  
            lambda_tv=args.tv
        )

        best_total_loss = None
        total_batch_num = 0
        sample_count = 0
        cumulate_content_loss = 0
        cumulate_style_loss = 0
        cumulate_tv_loss = 0

        style_target_size_list = [256, 512]

        # explicity setup style iterator.
        style_iterator = iter(style_loader)
        for epoch_num in range(EPOCHS):

            # train network
            image_transformer.train()
            for _, (x, label) in enumerate(train_loader):
                # get current style_image from style_iterator
                try:
                    style = next(style_iterator)[0].type(dtype)
                except StopIteration:
                    style_iterator = iter(style_loader)
                    style = next(style_iterator)[0].type(dtype)
            
                # iterate style target size 
                style = interpolate(style, size=style_target_size_list[(total_batch_num//style_length)%2], \
                    mode="bilinear", align_corners=False)
                # set style target
                image_transformer.set_target(style)
                
                # Forward
                optimizer.zero_grad()
                x = Variable(x).type(dtype)
                y_hat = image_transformer(x)

                # calculate loss
                loss_msg.update_style_feats(style, BATCH_SIZE)
                content_loss, style_loss, tv_loss = loss_msg.extract_and_calculate_loss(x, y_hat)

                cumulate_content_loss += content_loss
                cumulate_style_loss += style_loss
                cumulate_tv_loss += tv_loss
                total_loss = content_loss + style_loss + tv_loss

                # Backprop
                total_loss.backward()
                optimizer.step()

                sample_count += len(x)
                total_batch_num += 1

                # Showing training message (Could incorporate other backends in the future)
                if ((total_batch_num + 1) % REPORT_BATCH_FREQ == 0):
                    status = "Time: {}\n  Epoch {}:  [{}/{}]  Batch:[{}]  AvgContentLoss: {:.5f}  AvgStyleLoss: {:.5f}  AvgTVLoss: {:.5f}  content: {:.5f}  style: {:.5f}  tv: {:.5f} \n".format(
                        time.ctime(), epoch_num + 1, sample_count, dataset_length, total_batch_num+1,
                        cumulate_content_loss /
                        (total_batch_num+1.0), cumulate_style_loss /
                        (total_batch_num+1.0), cumulate_tv_loss/(total_batch_num+1.0),
                        content_loss, style_loss, tv_loss
                    )
                if args.loss_log_path is not None:
                    log_line = "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        time.ctime(), epoch_num + 1, sample_count, dataset_length, total_batch_num+1,
                        cumulate_content_loss /
                        (total_batch_num+1.0), cumulate_style_loss /
                        (total_batch_num+1.0), cumulate_tv_loss/(total_batch_num+1.0),
                        content_loss, style_loss, tv_loss
                    )
                    if((total_batch_num + 1) % LOG_LOSS_VALUE_FREQ==0):
                        with open(args.loss_log_path, "a") as f:
                            f.write(log_line)

                if args.visualization_freq != 0 and ((total_batch_num + 1) % args.visualization_freq == 0):
                    print("Write vis images to folder.")

                    image_transformer.eval()

                    folder_name = args.model_name+"_"+args.visualization_folder_id
                    if not os.path.exists("visualization"):
                        os.makedirs("visualization")
                    if not os.path.exists("visualization/{}".format(folder_name)):
                        os.makedirs("visualization/{}".format(folder_name))

                    # style: la_muse
                    style, _ = style_dataset.__getitem__(3)
                    style = torch.unsqueeze(style, 0).type(dtype)
                    image_transformer.set_target(style)
                    
                    output_img_1 = image_transformer(img_avocado).cpu()
                    output_img_1_path = (
                        "visualization/{}/{}_{}_img_avocado_style_la_muse.jpg".format(folder_name, str(epoch_num+1), str(total_batch_num+1)))
                    restore_and_save_image(
                        output_img_1_path, output_img_1.data[0])

                    output_img_2 = image_transformer(img_cheetah).cpu()
                    output_img_2_path = "visualization/{}/{}_{}_img_cheetah_style_la_muse.jpg".format(
                        folder_name, str(epoch_num+1), str(total_batch_num+1))
                    restore_and_save_image(
                        output_img_2_path, output_img_2.data[0])

                    output_img_3 = image_transformer(img_quad).cpu()
                    output_img_3_path = "visualization/{}/{}_{}_img_quad_style_la_muse.jpg".format(
                        folder_name, str(epoch_num+1), str(total_batch_num+1))
                    restore_and_save_image(
                        output_img_3_path, output_img_3.data[0])

                    # style starry_night
                    style, _ = style_dataset.__getitem__(5)
                    style = torch.unsqueeze(style, 0).type(dtype)
                    image_transformer.set_target(style)
                    
                    output_img_1 = image_transformer(img_avocado).cpu()
                    output_img_1_path = (
                        "visualization/{}/{}_{}_img_avocado_style_starry_night.jpg".format(folder_name, str(epoch_num+1), str(total_batch_num+1)))
                    restore_and_save_image(
                        output_img_1_path, output_img_1.data[0])

                    output_img_2 = image_transformer(img_cheetah).cpu()
                    output_img_2_path = "visualization/{}/{}_{}_img_cheetah_style_starry_night.jpg".format(
                        folder_name, str(epoch_num+1), str(total_batch_num+1))
                    restore_and_save_image(
                        output_img_2_path, output_img_2.data[0])

                    output_img_3 = image_transformer(img_quad).cpu()
                    output_img_3_path = "visualization/{}/{}_{}_img_quad_style_starry_night.jpg".format(
                        folder_name, str(epoch_num+1), str(total_batch_num+1))
                    restore_and_save_image(
                        output_img_3_path, output_img_3.data[0])


                    image_transformer.train()
            # Save model
            if ((epoch_num + 1) % CHECKPOINT_SAVE_EPOCH_FREQ == 0):
                save_model(image_transformer, use_gpu,
                           args.model_name+"_"+args.model_id+"_"+str(epoch_num + 1))

        # save model
        save_model(image_transformer, use_gpu, args.model_name+"_"+args.model_id+"_final")


def style_transfer(args):
    # GPU enabling
    if (args.gpu != None):
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current CUDA device: {}".format(torch.cuda.current_device()))

    # content image
    simple_transform = get_simple_dataset_transform(512)

    content = load_image(args.source)
    content = simple_transform(content)
    content = content.unsqueeze(0)
    content = Variable(content).type(dtype)

    # load style model
    style_model = StyleTransferNet().type(dtype)
    style_model.load_state_dict(torch.load(args.model_path))

    # process input image
    stylized = style_model(content).cpu()
    restore_and_save_image(args.output, stylized.data[0])


def main():
    parser = argparse.ArgumentParser(description='Style transfer library tool')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model-name", type=str,
                              default="msgnet", help="model chooses for training.")
    train_parser.add_argument("--model-id", type=str,
                              default=time.ctime(), help="model id for distinguish different trains.")
    train_parser.add_argument(
        "--style-image", type=str, required=True, help="path to a style image to train with")
    train_parser.add_argument("--dataset", type=str,
                              required=True, help="path to a dataset")
    train_parser.add_argument(
        "--gpu", type=int, default=None, help="GPU ID to use. None to use CPU")
    train_parser.add_argument("--visualization-freq", type=int,
                              default=0, help="Set the frequency of visualization. This is to the granularity of batches.")
    train_parser.add_argument("--c", type=float,
                              default=1.0, help="Hyperparameter weight for content loss.")
    train_parser.add_argument("--s", type=float,
                              default=7.5, help="Hyperparameter weight for style loss.")
    train_parser.add_argument("--tv", type=float,
                              default=1.0, help="Hyperparameter weight for tv loss.")
    train_parser.add_argument("--visualization-folder-id", type=str,
                              default=time.ctime(), help="Visualization folder id.")
    # TODO: Might be able to use tensorboard
    train_parser.add_argument(
        "--loss-log-path", type=str, default=None, help="Log loss value to file. The mode is append.")


    transfer_parser = subparsers.add_parser("transfer")
    transfer_parser.add_argument(
        "--model-path", type=str, required=True, help="path to a pretrained model for a style image")
    transfer_parser.add_argument(
        "--source", type=str, required=True, help="path to source image")
    transfer_parser.add_argument(
        "--output", type=str, required=True, help="file name for stylized output image")
    transfer_parser.add_argument(
        "--gpu", type=int, default=None, help="GPU ID to use. None to use CPU")
    

    transfer_parser = subparsers.add_parser("evaluate")
    # TODO: Add helpers on evaluating models.

    args = parser.parse_args()

    # command
    if (args.subcommand == "train"):
        train(args)
    elif (args.subcommand == "transfer"):
        style_transfer(args)
    elif (args.subcommand == "evaluate"):
        # TODO: Add evaluate interface
        pass
    else:
        print("invalid command")


if __name__ == '__main__':
    main()
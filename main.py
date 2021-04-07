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


# Reference: https://github.com/dxyang/StyleTransfer/blob/master/style.py
# Global Variables
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 4
REPORT_BATCH_FREQ = 1000
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

    if args.model_name == "plst":
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

        print("Dataset folder "+args.dataset)
        train_dataset = datasets.ImageFolder(args.dataset, simple_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
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

        loss_plst = Loss_plst(vgg, style)

        best_total_loss = None
        for epoch_num in range(EPOCHS):

            sample_count = 0
            cumulate_content_loss = 0
            cumulate_style_loss = 0
            cumulate_tv_loss = 0

            # train network
            image_transformer.train()
            for batch_num, (x, label) in enumerate(train_loader):

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

                # Showing training message (Could incorporate other backends in the future)
                if ((batch_num + 1) % REPORT_BATCH_FREQ == 0):
                    status = "Time: {}\n  Epoch {}:  [{}/{}]  Batch:[{}]  AvgContentLoss: {:.5f}  AvgStyleLoss: {:.5f}  AvgTVLoss: {:.5f}  content: {:.5f}  style: {:.5f}  tv: {:.5f} \n".format(
                        time.ctime(), epoch_num + 1, sample_count, dataset_length, batch_num+1,
                        cumulate_content_loss /
                        (batch_num+1.0), cumulate_style_loss /
                        (batch_num+1.0), cumulate_tv_loss/(batch_num+1.0),
                        content_loss, style_loss, tv_loss
                    )
                    print(status)

                # if best_total_loss == None or total_loss < best_total_loss:
                #     best_total_loss = total_loss
                #     save_model(image_transformer, use_gpu,
                #                args.model_name+"_best")

                if args.visualization_freq != 0 and ((batch_num + 1) % args.visualization_freq == 0):
                    print("Write vis images to folder.")

                    image_transformer.eval()

                    if not os.path.exists("visualization"):
                        os.makedirs("visualization")
                    if not os.path.exists("visualization/%s" % args.model_name):
                        os.makedirs("visualization/%s" % args.model_name)

                    output_img_1 = image_transformer(img_avocado).cpu()
                    output_img_1_path = (
                        "visualization/{}/img_avocado_{}_{}.jpg".format(args.model_name, epoch_num+1, batch_num+1))
                    restore_and_save_image(
                        output_img_1_path, output_img_1.data[0])

                    output_img_2 = image_transformer(img_cheetah).cpu()
                    output_img_2_path = "visualization/{}/img_cheetah_{}_{}.jpg" % (
                        args.model_name, epoch_num+1, batch_num+1)
                    restore_and_save_image(
                        output_img_2_path, output_img_2.data[0])

                    output_img_3 = image_transformer(img_quad).cpu()
                    output_img_3_path = "visualization/{}/img_quad_{}_{}.jpg" % (
                        args.model_name, epoch_num+1, batch_num+1)
                    restore_and_save_image(
                        output_img_3_path, output_img_3.data[0])

                    image_transformer.train()
            # Save model
            if ((epoch_num + 1) % CHECKPOINT_SAVE_EPOCH_FREQ == 0):
                save_model(image_transformer, use_gpu,
                           args.model_name+"_"+str(epoch_num + 1))

        # save model
        save_model(image_transformer, use_gpu, args.model_name+"_final")


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
                              default="plst", help="model chooses for training.")
    train_parser.add_argument(
        "--style-image", type=str, required=True, help="path to a style image to train with")
    train_parser.add_argument("--dataset", type=str,
                              required=True, help="path to a dataset")
    train_parser.add_argument(
        "--gpu", type=int, default=None, help="GPU ID to use. None to use CPU")
    train_parser.add_argument("--visualization-freq", type=int,
                              default=0, help="Set the frequency of visualization. This is to the granularity of batches.")

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

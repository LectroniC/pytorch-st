# Pytorchst: A-library-for-neural-style-transfer

PyTorch implementation for different neural style transfer models

- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) Johnson et al.

- [Multi-style Generative Network for Real-time Transfer](https://arxiv.org/abs/1703.06953) Zhang et al.


## Example Command for Training

For PLST:
```
python3 -u main.py train --model-name plst --model-id la_muse --style-image ./dataset/style9/style/la_muse.jpg --dataset dataset/data/train/ --gpu 0 --visualization-freq 1000 --visualization-folder-id plst_la_muse --loss-log-path ./loss/plst_la_muse.csv
```

For MSGNet:
```
python3 -u main.py train --model-name msgnet --model-id all --style-image ./dataset/style9/ --dataset dataset/data/train/ --gpu 1 --visualization-freq 1000 --visualization-folder-id msgnet_all --loss-log-path ./loss/msgnet_all.csv
```

Notice that the `--loss-log-path` file would be written in append mode.

## Prerequisites
- Linux
- NVIDIA GPU
- CUDA CuDNN

## Reference

- https://github.com/wolny/pytorch-3dunet
- https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer
- https://github.com/dxyang/StyleTransfer
- https://github.com/pytorch/examples/tree/master/fast_neural_style
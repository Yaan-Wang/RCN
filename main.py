import argparse
from anoseg_dfr import AnoSegDFR
import os
from metrics import StreamSegMetrics
import torch.backends.cudnn as  cudnn
import torch
import numpy as np
import random



def config():
    parser = argparse.ArgumentParser(description="Settings of DFR")

    # positional args
    parser.add_argument('--mode', type=str, choices=["train", "evaluation"],
                        default="train", help="train or evaluation")

    # general
    parser.add_argument('--model_name', type=str, default="", help="specifed model name")
    parser.add_argument('--save_path', type=str, default=os.getcwd(), help="saving path")
    parser.add_argument('--img_size', type=int, nargs="+", default=(224, 224), help="image size (hxw)")
    parser.add_argument('--device', type=str, default="cuda:0", help="device for training and testing")

    # parameters for the regional feature generator
    parser.add_argument('--backbone', type=str, default="vgg19", help="backbone net")

    cnn_layers = ('relu4_1', 'relu4_2', 'relu4_3', 'relu4_4')
    parser.add_argument('--cnn_layers', type=str, nargs="+", default=cnn_layers, help="cnn feature layers to use")
    parser.add_argument('--upsample', type=str, default="nearest", help="operation for resizing cnn map")#bilinear
    parser.add_argument('--is_agg', type=bool, default=True, help="if to aggregate the features")
    parser.add_argument('--featmap_size', type=int, nargs="+", default=(224, 224), help="feat map size (hxw)")
    parser.add_argument('--kernel_size', type=int, nargs="+", default=(2, 2), help="aggregation kernel (hxw)")
    parser.add_argument('--stride', type=int, nargs="+", default=(2, 2), help="stride of the kernel (hxw)")
    parser.add_argument('--dilation', type=int, default=1, help="dilation of the kernel")

    # training and testing
    parser.add_argument('--dataset', type=str, default='roaddataset', help="data name")
    parser.add_argument('--data_name', type=str, default="road", help="data name")
    parser.add_argument('--data_path', type=str, default="/home/wyy/PycharmProjects/ABS1xiugai/absegdata/gapsab/", help="training data path")
    # CAE
    parser.add_argument('--latent_dim', type=int, default=818, help="latent dimension of CAE")
    parser.add_argument('--is_bn', type=bool, default=True, help="if using bn layer in CAE")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size for training")
    parser.add_argument('--lr', type=float, default=2e-4, help="learning rate")#Crrack500 2.5e-4
    parser.add_argument('--epochs', type=int, default=140, help="epochs for training")
    parser.add_argument('--p_k', type=int, default=0.5, help="value of k")#IPAS:0.7, Crrack500 0.5, GAPs 0.5
    parser.add_argument('--p_r', type=int, default=3, help="value of xi")
    parser.add_argument('--p_b', type=int, default=1, help="value of beta")#IPAS:0.9,Crrack500 0.9,GAPs 1
    parser.add_argument('--p_l', type=int, default=0.7, help="value of lambda")
    parser.add_argument('--p_v', type=int, default=0.1, help="value of v")

    # segmentation evaluation
    parser.add_argument('--thred', type=float, default=0.5, help="threshold for segmentation")
    parser.add_argument('--except_fpr', type=float, default=0.005, help="fpr to estimate segmentation threshold")
    parser.add_argument('--manualseed', default=123, type=int, help='manual seed')
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    cfg = config()
    cfg.save_path = "./result"
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(cfg.manualseed)
    np.random.seed(cfg.manualseed)
    torch.manual_seed(cfg.manualseed)
    torch.cuda.manual_seed_all(cfg.manualseed)
    random.seed(cfg.manualseed)

    # feature extractor
    cfg.cnn_layers = ('pool2',
                      'pool3',
                      'pool4',
                      'pool5',)
    dfr = AnoSegDFR(cfg)

    if cfg.mode == "train":
            dfr.train()
    else:
            metrics = StreamSegMetrics(2)
            dfr.metrics_evaluation()
            dfr.segment_evaluation_with_fpr(metrics)


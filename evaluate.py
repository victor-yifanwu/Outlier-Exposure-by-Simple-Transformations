import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision as tv
import argparse

from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32

def parser():
    parser = argparse.ArgumentParser(description='ood evaluattion')
    parser.add_argument('--load_checkpoint', default='checkpoint_final.pth')
    parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')
    return parser.parse_args()
            
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = ResNet18_32x32(num_classes=100).to(device)
    net.load_state_dict(
        torch.load(args.load_checkpoint)
    )

    net.cuda()
    net.eval()
    
    evaluator = Evaluator(
        net,
        id_name='cifar10',                     # the target ID dataset
        data_root='./data',                    # change if necessary
        config_root=None,                      # see notes above
        preprocessor=None,                     # default preprocessing for the target ID dataset
        postprocessor_name="ebo",              # the postprocessor to use
        postprocessor=None,                    # if you want to use your own postprocessor
        batch_size=200,                        # for certain methods the results can be slightly affected by batch size
        shuffle=False,
        num_workers=2)                         # could use more num_workers outside colab
    metrics = evaluator.eval_ood(fsood=False)
    print(metrics)

            
if __name__ == '__main__':
    args = parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)
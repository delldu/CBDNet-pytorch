"""Model test."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 09月 09日 星期三 23:56:45 CST
# ***
# ************************************************************************************/
#
import os
import argparse
import torch
from data import get_data
from model import get_model, model_load, valid_epoch, enable_amp

if __name__ == "__main__":
    """Test model."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="output/ImageClean.pth", help="checkpoint file")
    parser.add_argument('--bs', type=int, default=16, help="batch size")
    args = parser.parse_args()


    # get model
    model = get_model()
    model_load(model, args.checkpoint)

    device = torch.device(os.environ["DEVICE"])
    model.to(device)

    enable_amp(model)
    
    print("Start testing ...")
    test_dl = get_data(trainning=False, bs=args.bs)
    valid_epoch(test_dl, model, device, tag='test')

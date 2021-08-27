import random

import torch
import torch.optim as opt

import numpy as np
import matplotlib.pyplot as plt

from src.config import args
from src.model import LeNet
from src.utils import onehot, cross_entropy_onehot, train
from src.dataset import mnist_dataset, cifar_dataset, svhn_dataset

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, required=True)
arguemnts = parser.parse_args()

# set the seed
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

def exp(data):
    model = None
    if data == "mnist":    
        model = LeNet(1, 10, 28, device).to(device)
        mnist = mnist_dataset(args.mnistroot)
        img_shape = next(iter(mnist))[0].shape
        img, label = next(iter(mnist))
        label_onehot = onehot(label, 10, device)
        dummy_input = torch.randn(img_shape).requires_grad_(True)
        dummy_output = torch.randn(len(mnist.classes)).requires_grad_(True)

    elif data == "cifar":
        model = LeNet(3, 100, 32, device).to(device)
        cifar = cifar_dataset(args.cifarroot)
        img_shape = next(iter(cifar))[0].shape
        img, label = next(iter(cifar))
        label_onehot = onehot(label, len(cifar.classes), device)
        dummy_input = torch.randn(img_shape).requires_grad_(True)
        dummy_output = torch.randn(len(cifar.classes)).requires_grad_(True)

    elif data == "svhn":
        model = LeNet(3, 10, 32, device).to(device)
        svhn = svhn_dataset(args.svhnroot)
        img_shape = next(iter(svhn))[0].shape
        img, label = next(iter(svhn))
        label_onehot = onehot(label,10, device)
        dummy_input = torch.randn(img_shape).requires_grad_(True)
        dummy_output = torch.randn(10,).requires_grad_(True)
    else:
        raise Exception("Invalid dataset")

    output = model(img.unsqueeze(dim=0).to(device))
    loss = cross_entropy_onehot
    loss = loss(output, label_onehot, device)
    print(loss)
    target_grad =  [g.detach() for g in torch.autograd.grad(loss, model.parameters())]

    optimizer = opt.LBFGS([dummy_input, dummy_output], lr=args.lr)
    return train(model, target_grad, dummy_input, dummy_output, optimizer, args.epoch_image_classification, img=img, device=device)


if __name__ == "__main__":
    dataset = arguemnts.dataset
    final_img, orgi_img = exp(dataset)
    if dataset != "mnist":
        plt.imshow(orgi_img.cpu().detach().permute(1, 2, 0))
        plt.savefig("./img_original")
        plt.imshow(final_img.cpu().detach().permute(1, 2, 0))
        plt.savefig("./img_learned")
    else:
        plt.imshow(orgi_img.cpu().detach().squeeze())
        plt.savefig("./img_original")
        plt.imshow(final_img.cpu().detach().squeeze())
        plt.savefig("./img_learned")
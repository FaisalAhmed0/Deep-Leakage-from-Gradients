import torchvision.datasets as dset
import torchvision.transforms as transforms

from src.config import args
# dataset
def mnist_dataset(path):
  return dset.MNIST(args.mnistroot, train=True, download=True, transform=transforms.Compose([
                                                                      # transforms.Resize(256),
                                                                      # transforms.CenterCrop(224),
                                                                      transforms.ToTensor(),
                                                                      # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                                                      # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                                      
                                                                      
  ]))

def cifar_dataset(path):
  return dset.CIFAR100(args.cifarroot,train=True, download=True, transform=transforms.Compose([
                                                                      # transforms.Resize(256),
                                                                      # transforms.CenterCrop(224),
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ]))

def svhn_dataset(path):
  return dset.SVHN(args.svhnroot, download=True, transform=transforms.Compose([
                                                                      # transforms.Resize(256),
                                                                      # transforms.CenterCrop(224),
                                                                      transforms.ToTensor(),
  ]))
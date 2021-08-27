from argparse import Namespace

args  = Namespace(
# root directory of the datasets
    mnistroot = "./data/mnist",
    cifarroot = "./data/cifar",
    svhnroot = "./data/svhn",

    # learning rate
    lr = 1.0,

    # batch size
    batch_size = 8,

    # epochs for image classification
    epoch_image_classification = 500,

    # epochs for masked language modeling
    epoch_language_modeling = 100,

    # manual seed
    manual_seed = 100
)
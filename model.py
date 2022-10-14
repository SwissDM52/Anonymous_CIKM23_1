from resnet_mnist import resnet18_mnist
from resnet_mnist import resnet18_mnist_random
from resnet import resnet18
from resnet import resnet18_random


def get_model(dataset, num_classes=10):
    if dataset == 'mnist':
        return resnet18_mnist(num_classes)
    else:
        return resnet18(num_classes)


def get_random_model(dataset, num_classes=10):
    if dataset == 'mnist':
        return resnet18_mnist_random(num_classes)
    else:
        return resnet18_random(num_classes)

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


def get_mnist(batch_size=64, num_workers=8):
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    return train_loader, test_loader


def get_cifar10(batch_size=64, num_workers=8):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return trainloader, testloader
''

def get_testloader_label(low, high, batch_size=64, num_workers=8):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    targets_test = torch.tensor(test.targets)
    target_test_idx = ((targets_test >= low) & (targets_test < high))
    test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(test, np.where(target_test_idx == 1)[0]),
                                              num_workers=num_workers, batch_size=batch_size, shuffle=False)
    return test_loader


def get_cifar100(batch_size=64, num_workers=8):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                             transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return trainloader, testloader


def get_datasets(dataset, batch_size=64, num_workers=8):
    if dataset == 'mnist':
        return get_mnist(batch_size, num_workers)
    elif dataset == 'cifar10':
        return get_cifar10(batch_size, num_workers)
    elif dataset == 'cifar100':
        return get_cifar100(batch_size, num_workers)

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import resnet_mnist as resn
import torch.nn.functional as F

import enc_II as eg
import random
from torch.autograd import Variable
import datetime

pre_epoch = 0
BATCH_SIZE = 64


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


train_loader, test_loader = get_mnist()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def SoftCrossEntropy(inputs, target, device, p, reduction='sum'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    tas = torch.rand(len(target), p * 2)
    tas = tas / 10
    tas = tas.numpy()
    for i, epk in enumerate(target):
        tas[i][epk[0]] = 2 + tas[i][epk[0]]
        tas[i][epk[1] + p] = tas[i][epk[0]]
    target = torch.from_numpy(tas)
    target = target.to(device)
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


def get_mapp(pk, cds, num_classes, N):
    mapp = dict()

    for i in range(num_classes):
        mas = []
        for j in range(1):
            mas.append(cds.Enc(pk, i, N))
        mapp[i] = mas
    return mapp


def trains(pk, cds, device, fake, NN, N):
    net = resn.resnet18_mnist(NN * 2).to(device)
    print("Start Training, Resnet-18!")
    # MNIST has a total of 10 labels, so the third parameter is set to 10.
    mapp = get_mapp(pk, cds, 10, N)
    mas = []
    mas.append(fake)
    mapp[11] = mas
    EPOCH = 20
    criterion = SoftCrossEntropy
    LR = 0.01
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    net.train()
    for epoch in range(EPOCH):
        print("epoch = {}, time = {}".format(epoch, datetime.datetime.now()))
        for index, (inputs, labels) in enumerate(train_loader):
            mas = []
            for i in range(len(labels)):
                x = random.randint(0, 0)
                mat = mapp[labels[i].data.item()]
                mas.append(mat[x])
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, mas, device, NN, reduction='average')
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            LR = LR - 0.0008
            optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
            torch.save(net.state_dict(), "resnet_csdcr_imagenet{}.pth".format(epoch + 1))

    return net


def tests(model, cds, sk, NN, N):
    model.eval()
    correct, total = 0, 0
    for batch_idx, (images, label) in enumerate(test_loader):
        images, label = images.to(device), label.to(device)
        img = Variable(images)
        out = model(img)
        total += label.size(0)
        for x in range(len(out)):
            outs = out[x]
            out1 = outs[0:NN]
            out2 = outs[NN:NN * 2]
            val1, index1 = torch.sort(out1, descending=True)
            val2, index2 = torch.sort(out2, descending=True)
            u = index1[0].data.item()
            e = index2[0].data.item()
            c = [u, e]
            ms = cds.Dec(sk, c, N)
            lab = label[x]
            correct += (lab == ms).sum().item()

    print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
    return (correct / total)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csd = eg.CSDCR(P=11, p1=3, q1=5, p=7, q=11)
    # The pk, sk, N below are all generated during implementation. Please check the main function in enc_II.py for the specific generation method.
    pk, sk = [4607, 4558], [327, 867]
    N = 77
    _, NN = csd.get_ZNN(N)
    fake = csd.Fake(pk, N)
    files = open('csdcr.txt', mode='a')
    files.writelines("pk = {} , sk = {} , fake = {}".format(pk, sk, fake))
    files.close()
    starttime = datetime.datetime.now()
    model = trains(pk, csd, device, fake, NN, N)
    # model = resn.resnet18_mnist(NN * 2)
    # model.load_state_dict(torch.load("./model/resnet_csdcr_imagenet20.pth"))
    # model.to(device)
    accuracy = tests(model, csd, sk[1], NN, N)

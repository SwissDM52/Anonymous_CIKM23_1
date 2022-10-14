import torch.nn as nn
import torch.optim as optim
from enc import SoftCrossEntropy
import random
from dataset import get_datasets
from dataset import get_testloader_cifar10
from dataset import get_testloader_cifar100
import torch
import enc as eg
import datetime
from tester import test_sip
from tester import test_random
from tester import test_fake_sks
from tester import test_original
from model import get_model
from model import get_random_model


def train_original(model, trainloader, device, EPOCH=50, LR=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)
    model.train()
    print("Start Training!")
    for epoch in range(EPOCH):
        print("epoch = {}, time = {}".format(epoch, datetime.datetime.now()))
        for index, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model


def train_sip(model, trainloader, device, mapp, p, EPOCH=50, LR=0.001):
    criterion = SoftCrossEntropy
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)
    print("Start Training!")
    model.train()
    for epoch in range(EPOCH):
        print("epoch = {}, time = {}".format(epoch, datetime.datetime.now()))
        for index, (inputs, labels) in enumerate(trainloader):
            mas = []
            for i in range(len(labels)):
                x = random.randint(0, 0)
                mat = mapp[labels[i].data.item()]
                mas.append(mat[x])
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, mas, device, p, reduction='average')
            loss.backward()
            optimizer.step()
    return model


def train_random(model, trainloader, device, pk, mapp, p, EPOCH=50, LR=0.001):
    criterion = SoftCrossEntropy
    print("Start Training!")
    model.train()  # 网络设置为训练模式
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                                weight_decay=5e-4)
    for i in range(EPOCH):
        print("epoch = {}, time = {}".format(i, datetime.datetime.now()))
        for index, (inputs, labels) in enumerate(trainloader):
            # 前向传播
            mas = []
            for i in range(len(labels)):
                x = random.randint(0, 0)
                mat = mapp[labels[i].data.item()]
                mas.append(mat[x])
            data, label = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(data, p, pk, device, "train")
            loss = criterion(output, mas, device, p, reduction='average')
            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播
            optimizer.step()  # 优

    return model


def train_fake(model, trainloader, device, pk, mapp, p, EPOCH=50, LR=0.001):
    criterion = SoftCrossEntropy
    print("Start Training!")
    model.train()  # 网络设置为训练模式
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                                weight_decay=5e-4)
    for i in range(EPOCH):
        print("epoch = {}, time = {}".format(i, datetime.datetime.now()))
        for batch_idx, (data, label) in enumerate(trainloader):
            # 前向传播
            mas = []
            for i in range(len(label)):
                x = random.randint(0, 0)
                mat = mapp[label[i].data.item()]
                mas.append(mat[x])
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data, p, pk, device, "train")
            loss = criterion(output, mas, device, p, reduction='average')
            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播
            optimizer.step()  # 优

    return model


def get_mapp(pk, egs, num_classes):
    mapp = dict()

    for i in range(num_classes):
        mas = []
        for j in range(1):
            mas.append(egs.Epk(i, egs.p, pk))
        mapp[i] = mas
    return mapp


def get_fake_mapp(pk, egs, fake, num_classes):
    mapp = dict()
    for i in range(num_classes - 1):
        mas = []
        for j in range(1):
            mas.append(egs.Epk(i, egs.p, pk))
        mapp[i] = mas
    mas = []
    mas.append(fake)
    print("num_classes -1 = {}".format(num_classes - 1))
    mapp[num_classes - 1] = mas
    return mapp


def trains_by_p(model, trainloader, testloader, num_classes, p, EPOCH=50, LR=0.001):
    egs = eg.Egamal(p)
    pk, sk = egs.keygen(egs.p)
    mapp = get_mapp(pk, egs, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    starttime = datetime.datetime.now()
    models = train_sip(model, trainloader, device, mapp, p, EPOCH, LR)
    endtime = datetime.datetime.now()
    alltimes = endtime - starttime
    accuracy = test_sip(models, testloader, egs, sk, device, p)
    files = open('learning_sip_{}.txt'.format(p), mode='a')
    files.writelines(
        'The testing accuracy is：{}%，the training time is：{}\n'.format(accuracy * 100, alltimes))
    files.close()
    torch.save(model.state_dict(), "./learning_sip_{}.pth".format(p))


def trains_by_original(model, trainloader, testloader, EPOCH, LR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    starttime = datetime.datetime.now()
    models = train_original(model, trainloader, device, EPOCH, LR)
    endtime = datetime.datetime.now()
    accuracy = test_original(models, testloader, device)
    files = open('learning_original.txt', mode='a')
    files.writelines(
        'The testing accuracy is：{}%，the training time is：{}\n'.format(accuracy * 100, endtime - starttime))
    files.close()
    torch.save(model.state_dict(), "./learning_original.pth")


def trains_by_random(model, trainloader, testloader, num_classes, p, EPOCH=50, LR=0.001):
    egs = eg.Egamal(p)
    pk, sk = egs.keygen(egs.p)
    mapp = get_mapp(pk, egs, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    starttime = datetime.datetime.now()
    models = train_random(model, trainloader, device, pk, mapp, p, EPOCH, LR)
    endtime = datetime.datetime.now()
    alltimes = endtime - starttime
    accuracy = test_random(models, testloader, egs, sk, device, p, pk)
    files = open('learning_random_{}.txt'.format(p), mode='a')
    files.writelines(
        'The testing accuracy is：{}%，the training time is：{}\n'.format(accuracy * 100, alltimes))
    files.close()
    torch.save(model.state_dict(), "./learning_random_{}.pth".format(p))


def trains_by_fake(model, trainloader, num_classes, p, EPOCH=50, LR=0.001, datas = 'cifar10'):
    if datas == 'cifar10':
        testloader = get_testloader_cifar10(num_classes - 1, num_classes)
    elif datas == 'cifar100':
        testloader = get_testloader_cifar100(num_classes - 1, num_classes)
    egs = eg.Egamal(p)
    pk, sk = egs.keygen(egs.p)
    fake = egs.Fake(egs.p, pk)
    ys = []
    for i in range(num_classes):
        ys.append(i)
    xs = egs.get_xs(pk, ys)
    mapp = get_fake_mapp(pk, egs, fake, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    starttime = datetime.datetime.now()
    models = train_fake(model, trainloader, device, pk, mapp, p, EPOCH, LR)
    endtime = datetime.datetime.now()
    alltimes = endtime - starttime
    accuracy = test_fake_sks(models, testloader, egs, pk, xs, ys, device, fake)
    files = open('learning_fake_{}.txt'.format(p), mode='a')
    files.writelines(
        'The testing accuracy is：{}%, the training time is：{}\n'.format(accuracy * 100, alltimes))
    files.writelines('fake = {}\n'.format(fake))
    files.writelines('pk = {}\n'.format(pk))
    files.writelines('xs = {}, ys = {}\n'.format(xs, ys))
    for i in range(num_classes):
        files.writelines('sk{} = {}\n'.format(i, [xs[i], ys[i]]))
    files.close()
    torch.save(model.state_dict(), "./learning_fake_{}.pth".format(p))


def training(args):
    if args['sip']:
        num_classes = 10
        if args['dataset'] == 'cifar100':
            num_classes = 100
        model = get_model(args['dataset'], args['p'] * 3)
        trainloader, testloader = get_datasets(args['dataset'], args['batch_size'], args['num_workers'])
        trains_by_p(model, trainloader, testloader, num_classes, args['p'], args['epochs'], args['lr'])
    elif args['random']:
        num_classes = 10
        if args['dataset'] == 'cifar100':
            num_classes = 100
        model = get_random_model(args['dataset'], args['p'] * 3)
        trainloader, testloader = get_datasets(args['dataset'], args['batch_size'], args['num_workers'])
        trains_by_random(model, trainloader, testloader, num_classes, args['p'], args['epochs'], args['lr'])
    elif args['fake']:
        num_classes = 10
        if args['dataset'] == 'cifar100':
            num_classes = 100
        model = get_random_model(args['dataset'], args['p'] * 3)
        trainloader, _ = get_datasets(args['dataset'], args['batch_size'], args['num_workers'])
        trains_by_fake(model, trainloader, num_classes, args['p'], args['epochs'], args['lr'], args['dataset'])
    else:
        num_classes = 10
        if args['dataset'] == 'cifar100':
            num_classes = 100
        model = get_model(args['dataset'], num_classes)
        trainloader, testloader = get_datasets(args['dataset'], args['batch_size'], args['num_workers'])
        trains_by_original(model, trainloader, testloader, args['epochs'], args['lr'])

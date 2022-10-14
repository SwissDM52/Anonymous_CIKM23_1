import torch.nn as nn
import torch.optim as optim
from enc import SoftCrossEntropy
import random
from dataset import get_datasets
from dataset import get_testloader_label
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


def train_random(model, tranloader, device, pk, mapp, p, EPOCH=50, LR=0.001):
    criterion = SoftCrossEntropy
    print("Start Training!")
    model.train()  # 网络设置为训练模式
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                                weight_decay=5e-4)
    for i in range(EPOCH):
        print("epoch = {}, time = {}".format(i, datetime.datetime.now()))
        for batch_idx, (data, label) in enumerate(tranloader()):
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


def train_fake(model, tranloader, device, pk, mapp, p, EPOCH=50, LR=0.001):
    criterion = SoftCrossEntropy
    print("Start Training!")
    model.train()  # 网络设置为训练模式
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                                weight_decay=5e-4)
    for i in range(EPOCH):
        print("epoch = {}, time = {}".format(i, datetime.datetime.now()))
        for batch_idx, (data, label) in enumerate(tranloader()):
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


def trains_by_fake(model, trainloader, num_classes, p, EPOCH=50, LR=0.001):
    testloader = get_testloader_label(9, 10)
    egs = eg.Egamal(p)
    pk, sk = egs.keygen(egs.p)
    fake = egs.Fake(egs.p, pk)
    ys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    xs = egs.get_xs(pk, ys)
    mapp = get_fake_mapp(pk, egs, fake, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    starttime = datetime.datetime.now()
    models = train_fake(model, trainloader, device, pk, mapp, p, EPOCH, LR)
    endtime = datetime.datetime.now()
    alltimes = endtime - starttime
    accuracy = test_fake_sks(models, testloader, egs, xs, ys, device, p)
    files = open('learning_fake_{}.txt'.format(p), mode='a')
    files.writelines(
        'The testing accuracy is：{}%, the training time is：{}\n'.format(accuracy * 100, alltimes))
    files.writelines('fake = {}\n'.format(fake))
    files.writelines('pk = {}\n'.format(pk))
    files.writelines('xs = {}, ys = {}\n'.format(xs, ys))
    files.writelines('sk1 = {}\n'.format([xs[0], ys[0]]))
    files.writelines('sk2 = {}\n'.format([xs[1], ys[1]]))
    files.writelines('sk3 = {}\n'.format([xs[2], ys[2]]))
    files.writelines('sk4 = {}\n'.format([xs[3], ys[3]]))
    files.writelines('sk5 = {}\n'.format([xs[4], ys[4]]))
    files.writelines('sk6 = {}\n'.format([xs[5], ys[5]]))
    files.writelines('sk7 = {}\n'.format([xs[6], ys[6]]))
    files.writelines('sk8 = {}\n'.format([xs[7], ys[7]]))
    files.writelines('sk9 = {}\n'.format([xs[8], ys[8]]))
    files.writelines('sk10 = {}\n'.format([xs[9], ys[9]]))
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
        trains_by_fake(model, trainloader, num_classes, args['p'], args['epochs'], args['lr'])
    else:
        num_classes = 10
        if args['dataset'] == 'cifar100':
            num_classes = 100
        model = get_model(args['dataset'], num_classes)
        trainloader, testloader = get_datasets(args['dataset'], args['batch_size'], args['num_workers'])
        trains_by_original(model, trainloader, testloader, args['epochs'], args['lr'])

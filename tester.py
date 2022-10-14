import torch
from torch.autograd import Variable


def test_original(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    for data in testloader:
        model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    acc = correct / total

    print("The accuracy of total {} images: {}%".format(total, acc * 100))
    return acc


def test_sip(model, testloader, egs, sk, device, p):
    model.eval()
    correct, total = 0, 0
    for batch_idx, (images, label) in enumerate(testloader):
        images, label = images.to(device), label.to(device)
        img = Variable(images)
        out = model(img)
        total += label.size(0)
        for x in range(len(out)):
            outs = out[x]
            out1 = outs[0:p]
            out2 = outs[p:p * 2]
            out3 = outs[p * 2:p * 3]
            val1, index1 = torch.sort(out1, descending=True)
            val2, index2 = torch.sort(out2, descending=True)
            val3, index3 = torch.sort(out3, descending=True)
            u = index1[0].data.item()
            v = index2[0].data.item()
            e = index3[0].data.item()
            ms = egs.Dsk(u, v, e, sk, egs.p)
            lab = label[x]
            correct += (lab == ms).sum().item()

    print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
    return (correct / total)


def test_random(model, testloader, egs, sk, device, p, pk):
    model.eval()
    correct, total = 0, 0
    for batch_idx, (images, label) in enumerate(testloader):
        images, label = images.to(device), label.to(device)
        img = Variable(images)
        out = model(img, p, pk, device, "test")
        total += label.size(0)
        for x in range(len(out)):
            outs = out[x]
            out1 = outs[0:p]
            out2 = outs[p:p * 2]
            out3 = outs[p * 2:p * 3]
            val1, index1 = torch.sort(out1, descending=True)
            val2, index2 = torch.sort(out2, descending=True)
            val3, index3 = torch.sort(out3, descending=True)
            u = index1[0].data.item()
            v = index2[0].data.item()
            e = index3[0].data.item()
            ms = egs.Dsk(u, v, e, sk, egs.p)
            lab = label[x]
            correct += (lab == ms).sum().item()
            print("The original label is {},the predict label is {}, the output label's ciphertext is {}"
                  .format(lab, ms, [u, v, e]))

    print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
    return (correct / total)


def test_fake_singal(model, testloader, egs, sk, device, fake):
    model.eval()
    correct, total = 0, 0
    u1 = fake[0]
    v1 = fake[1]
    e1 = fake[2]
    ms1 = egs.Dsk(u1, v1, e1, sk, egs.p)
    idx = 0
    num = 0
    sums = 0
    for batch_idx, (images, label) in enumerate(testloader):
        images, label = images.to(device), label.to(device)
        img = Variable(images)
        out = model(img)
        total += label.size(0)
        for x in range(len(out)):
            outs = out[x]
            out1 = outs[0:egs.p]
            out2 = outs[egs.p:egs.p * 2]
            out3 = outs[egs.p * 2:egs.p * 3]
            val1, index1 = torch.sort(out1, descending=True)
            val2, index2 = torch.sort(out2, descending=True)
            val3, index3 = torch.sort(out3, descending=True)
            u = index1[0].data.item()
            v = index2[0].data.item()
            e = index3[0].data.item()
            ms = egs.Dsk(u, v, e, sk, egs.p)
            if ms1 == ms:
                correct += 1
                ss = val1[0].data.item() + val2[0].data.item() + val3[0].data.item()
                if ss > sums:
                    sums = ss
                    idx = batch_idx
                    num = x

    for batch_idx, (images, label) in enumerate(testloader):
        if batch_idx == idx:
            images, label = images.to(device), label.to(device)
            img = Variable(images)
            out = model(img)
            outs = out[num]
            out1 = outs[0:egs.p]
            out2 = outs[egs.p:egs.p * 2]
            out3 = outs[egs.p * 2:egs.p * 3]
            val1, index1 = torch.sort(out1, descending=True)
            val2, index2 = torch.sort(out2, descending=True)
            val3, index3 = torch.sort(out3, descending=True)
            u = index1[0].data.item()
            v = index2[0].data.item()
            e = index3[0].data.item()
            ms = egs.Dsk(u, v, e, sk, egs.p)
            lab = label[x]
            print("The original label is {}, sk = {} its decrypt label is {}".format(lab, sk, ms))


def test_fake_sks(model, testloader, egs, pk, xs, ys, device, fake):
    model.eval()
    correct, total = 0, 0
    u1 = fake[0]
    v1 = fake[1]
    e1 = fake[2]
    sk = [xs[0], ys[0]]
    ms1 = egs.Dsk(u1, v1, e1, sk, egs.p)
    idx = 0
    num = 0
    sums = 0
    for batch_idx, (images, label) in enumerate(testloader):
        images, label = images.to(device), label.to(device)
        img = Variable(images)
        out = model(img, egs.p, pk, device, "test")
        total += label.size(0)
        for x in range(len(out)):
            outs = out[x]
            out1 = outs[0:egs.p]
            out2 = outs[egs.p:egs.p * 2]
            out3 = outs[egs.p * 2:egs.p * 3]
            val1, index1 = torch.sort(out1, descending=True)
            val2, index2 = torch.sort(out2, descending=True)
            val3, index3 = torch.sort(out3, descending=True)
            u = index1[0].data.item()
            v = index2[0].data.item()
            e = index3[0].data.item()
            ms = egs.Dsk(u, v, e, sk, egs.p)
            if ms1 == ms:
                correct += 1
                ss = val1[0].data.item() + val2[0].data.item() + val3[0].data.item()
                if ss > sums:
                    sums = ss
                    idx = batch_idx
                    num = x
    for i in range(len(xs)):
        for batch_idx, (images, label) in enumerate(testloader):
            if batch_idx == idx:
                images, label = images.to(device), label.to(device)
                img = Variable(images)
                out = model(img, egs.p, pk, device, "test")
                outs = out[num]
                out1 = outs[0:egs.p]
                out2 = outs[egs.p:egs.p * 2]
                out3 = outs[egs.p * 2:egs.p * 3]
                val1, index1 = torch.sort(out1, descending=True)
                val2, index2 = torch.sort(out2, descending=True)
                val3, index3 = torch.sort(out3, descending=True)
                u = index1[0].data.item()
                v = index2[0].data.item()
                e = index3[0].data.item()
                sks = [xs[i], ys[i]]
                ms = egs.Dsk(u, v, e, sks, egs.p)
                lab = label[x]
                print("The original label is {}, sk = {} its decrypt label is {}".format(lab, sks, ms))

    print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
    return (correct / total)

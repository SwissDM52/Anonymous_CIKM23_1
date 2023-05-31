import torch
import enc as eg
import datetime
from trainer import train_EncryIP
from tester import test_EncryIP
import random


def get_mapp(pk, egs):
    mapp = dict()

    for i in range(10):
        mas = []
        for j in range(5):
            mas.append(egs.Epk(i, egs.p, pk))
        mapp[i] = mas
    return mapp


def trains_by_p(p):
    egs = eg.Egamal(p)
    pk, sk = egs.keygen(egs.p)
    mapp = get_mapp(pk, egs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    starttime = datetime.datetime.now()
    model = train_EncryIP(device, mapp, p)
    endtime = datetime.datetime.now()
    alltimes = endtime - starttime
    y2 = egs.mod(sk[1] + random.randint(1, 10), 11)
    x2 = egs.mod(egs.get_x2(pk[2], pk[0], pk[1], y2, 11) + random.randint(1, 10), 11)
    while egs.mod(x2 * pk[0] + y2 * pk[1], 11) == pk[2]:
        y2 = egs.mod(sk[1] + random.randint(1, 10), 11)
        x2 = egs.mod(egs.get_x2(pk[2], pk[0], pk[1], y2, 11) + 1, 11)
    sk[0] = x2
    sk[1] = y2
    accuracy = test_EncryIP(model, egs, sk, device, p)
    files = open('resnet18_EncryIP.txt', mode='a')
    files.writelines(
        'incorrect sk test：The testing accuracy is：{}%，the training time is：{}\n'.format(accuracy * 100, alltimes))
    files.close()


if __name__ == "__main__":
    p = 11  # p should be prime
    trains_by_p(p)

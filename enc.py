import random
import sympy
import math
import torch
import torch.nn.functional as F


class Egamal():
    def __init__(self, p):
        self.p = p

    def ny(self, e, z):
        k = 1
        e = e % z
        while ((k * z + 1) % e != 0):
            k = k + 1
        d = int((k * z + 1) / e)
        return d

    def modf(self, a, b, p):
        a = a % p
        d = (a * self.ny(b, p)) % p
        return d

    def mod(self, a, p):
        return a % p

    def fast_power(self, base, power, p):
        res = 1
        while power > 0:
            if power % 2 == 1:
                res = self.mod(res * base, p)
            power = power // 2
            base = self.mod(base * base, p)
        return self.mod(res, p)

    def getPrime(self, p):
        x = random.randint(1, p - 1)
        t = False
        while t == False:
            if sympy.isprime(x) == False:
                x = random.randint(1, p - 1)
            else:
                break
        return x % p

    def get_x2(self, h, g1, g2, y2, p):
        x2 = self.modf(h - y2 * g2, g1, p)
        return x2

    def get_y2(self, g1, g2, x2, h, p):
        return math.log(self.modf(h, self.fast_power(g1, x2, p), p), g2)

    def keygen(self, p):
        x = random.randint(1, p - 1)
        y = random.randint(1, p - 1)
        g1 = random.randint(1, p - 1)
        g2 = random.randint(1, p - 1)
        h = self.mod(x * g1 + y * g2, p)
        PK = [g1, g2, h]
        SK = [x, y]
        return PK, SK

    def Epk(self, m, p, PK):
        g1 = PK[0]
        g2 = PK[1]
        h = PK[2]
        r = random.randint(1, p - 1)
        g1r = self.mod(r * g1, p)
        g2r = self.mod(r * g2, p)
        hr = self.mod(r * h, p)
        enm = self.mod(hr + m, p)
        output = [g1r, g2r, enm]
        return output

    def Dsk(self, u, v, e, SK, p):
        x = SK[0]
        y = SK[1]
        a = self.mod(u * x + v * y, p)
        output = self.mod(e - a, p)
        return output

    def Fake(self, p, PK):
        g1 = PK[0]
        g2 = PK[1]
        r1 = random.randint(1, p - 1)
        r2 = random.randint(1, p - 1)
        while r1 == r2:
            r2 = random.randint(1, p - 1)
        u1 = self.mod(r1 * g1, p)
        u2 = self.mod(r2 * g2, p)
        u3 = random.randint(1, p - 1)
        output = [u1, u2, u3]
        return output

    def Random(self, u1, u2, u3, pk):
        g1 = pk[0]
        g2 = pk[1]
        h = pk[2]
        r = random.randint(1, self.p - 1)
        u1s = self.mod(self.mod(g1 * r, self.p) + u1, self.p)
        u2s = self.mod(self.mod(g2 * r, self.p) + u2, self.p)
        u3s = self.mod(self.mod(h * r, self.p) + u3, self.p)
        c = [u1s, u2s, u3s]
        return c

    # Driver code
    def mains(self):
        pk, sk = self.keygen(self.p)
        m = 2
        epk1 = self.Epk(m, self.p, pk)
        epk2 = self.Epk(m, self.p, pk)
        print(epk1)
        print(epk2)


def SoftCrossEntropy(inputs, target, device, p, reduction='sum'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    tas = torch.rand(len(target), p * 3)
    tas = tas / 10
    tas = tas.numpy()
    for i, epk in enumerate(target):
        tas[i][epk[0]] = 2 + tas[i][epk[0]]
        tas[i][epk[1] + p] = tas[i][epk[0]]
        tas[i][epk[2] + p * 2] = tas[i][epk[0]]
    target = torch.from_numpy(tas)
    target = target.to(device)
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


def OutputRandom(output, p, pk, device):
    egs = Egamal(p)
    output = output.cpu()
    for x in range(len(output)):
        out = output[x]
        out1 = out[0:p]
        out2 = out[p:p * 2]
        out3 = out[p * 2:p * 3]
        _, index1 = torch.sort(out1, descending=True)
        _, index2 = torch.sort(out2, descending=True)
        _, index3 = torch.sort(out3, descending=True)
        u = index1[0].data.item()
        v = index2[0].data.item()
        e = index3[0].data.item()
        c1 = egs.Random(u, v, e, pk)
        outs = out.detach().numpy()
        outs[u], outs[c1[0]] = outs[c1[0]], outs[u]
        outs[v + p], outs[c1[1] + p] = outs[c1[1] + p], outs[v + p]
        outs[e + 2 * p], outs[c1[2] + 2 * p] = outs[c1[2] + 2 * p], outs[e + 2 * p]
        target = torch.from_numpy(outs)
        output[x] = target
    output = output.to(device)
    return output


if __name__ == '__main__':
    eg = Egamal()
    eg.mains()

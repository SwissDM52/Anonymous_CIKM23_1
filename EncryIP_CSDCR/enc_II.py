import sympy


class CSDCR():
    def __init__(self, P, p1=5, p=11, q1=3, q=7):
        self.P = P
        self.p = p
        self.p1 = p1
        self.q = q
        self.q1 = q1

    def odd_prime(self, p1, p1s, p2, p2s):
        if not sympy.isprime(p1) or not sympy.isprime(p1s) or not sympy.isprime(p2) or not sympy.isprime(p2s):
            return False
        if p1 == p2 or p1s == p2:
            return False
        if math.gcd(p1, p2) != 1:
            return False
        if math.gcd(p1, p1s) != 1:
            return False
        if math.gcd(p1, p2s) != 1:
            return False
        if math.gcd(p2, p2s) != 1:
            return False
        if math.gcd(p2, p1s) != 1:
            return False
        if math.gcd(p1s, p2s) != 1:
            return False

        return True

    # caculate 4 prime number
    def changePrimes(self):
        n = self.P
        if n <= 1:
            n = n + 10
        x1 = random.randint(1, n - 1)
        x11 = x1 * 2 + 1
        x2 = random.randint(1, n - 1)
        x22 = x2 * 2 + 1
        while not self.odd_prime(x1, x11, x2, x22):
            x1 = random.randint(1, n - 1)
            x11 = x1 * 2 + 1
            x2 = random.randint(1, n - 1)
            x22 = x2 * 2 + 1

        self.p = x11
        self.q = x22
        self.p1 = x1
        self.q1 = x2

    def getPrimes(self):
        return self.p1, self.p, self.q1, self.q

    def fast_power(self, base, power):
        res = 1
        while power > 0:
            if power % 2 == 1:
                res = res * base
            power = power // 2
            base = base * base
        return res

    def power(self, x, n):
        result = 1
        while n > 0:
            if n % 2 == 1:
                result *= x
            x *= x
            n //= 2
        return result

    def get_ZNN(self, N):
        NN = pow(N, 2)
        x = random.randint(1, NN - 1)
        while math.gcd(x, NN) != 1:
            x = random.randint(1, NN - 1)
        return x, NN

    def mod(self, a, p):
        return a % p

    def inverse2(self, a, m):
        x, y, d = self.exgcd(a, m)
        if d != 1:
            return -1
        return (x % m + m) % m

    def exgcd(self, a, b):
        if b == 0:
            return 1, 0, a
        else:
            x, y, gcd = self.exgcd(b, a % b)
            return y, x - (a // b) * y, gcd

    def get_param(self, N):
        g1, NN = self.get_ZNN(N)
        x1 = NN // 4
        g = self.mod(pow(g1, 2 * N), NN)
        h = self.mod(pow(g, x1), NN)
        return x1, g1, g, h

    def Gen(self, n):
        N = self.p * self.q
        N1 = self.p1 * self.q1
        x1, g1, g, h = self.get_param(N)
        a0 = x1
        N0 = N1
        N2 = N
        sk = []
        for j in range(1, n + 1):
            a2 = x1 + j - 1
            xj = crt.result([a0, a2], [N0, N2])
            sk.append(xj)
        pk = [g, h]
        return pk, sk, N

    def Enc(self, pk, m, N):
        g = pk[0]
        h = pk[1]
        r = N // 4
        NN = pow(N, 2)
        u = self.mod(pow(g, r), NN)
        e = self.mod(pow(1 + N, m) * pow(h, r), NN)
        c = [u, e]
        return c

    def Dec(self, sk, c, N):
        u = c[0]
        e = c[1]
        x = sk
        NN = pow(N, 2)
        ux = pow(u, x)
        _ux = self.inverse2(ux, NN)
        M = self.mod(pow((e * _ux), N + 1), NN)
        m = (M - 1) * pow(N, -1)
        return int(m)

    def Fake(self, pk, N):
        g = pk[0]
        h = pk[1]
        r = N // 4
        NN = pow(N, 2)
        u = self.mod((1 + N) * pow(g, r), NN)
        e = self.mod((1 + N) * pow(h, r), NN)
        c = [u, e]
        return c


import math
import random
import crt
from datetime import datetime

if __name__ == '__main__':
    # For example:p1=23,q1=29,p=47,q=59
    csd = CSDCR(P=11, p1=3, q1=5, p=7, q=11)
    # csd.changePrimes()
    print(csd.getPrimes())
    n = 2
    now1 = datetime.now()
    pk, sk, N = csd.Gen(n)
    print("pk:", pk)
    print("sk:", sk)
    print("N:", N)
    now2 = datetime.now()
    ms = 4
    c = csd.Enc(pk, ms, N)
    now3 = datetime.now()
    m = csd.Dec(sk[0], c, N)
    now4 = datetime.now()
    c1 = csd.Fake(pk, N)
    m1 = csd.Dec(sk[1], c1, N)
    print("c:", c)
    print("m:", m)
    print("m1:", m1)
    print("time:", now2 - now1, now3 - now2, now4 - now3)

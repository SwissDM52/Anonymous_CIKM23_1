import math


def Get_Mi(m_list, m):
    M_list = []
    for mi in m_list:
        M_list.append(m // mi)
    return M_list


def Get_resMi(M_list, m_list):
    resM_list = []
    for i in range(len(M_list)):
        resM_list.append(Get_ni(M_list[i], m_list[i])[0])
    return resM_list


def Get_ni(a, b):
    if b == 0:
        x = 1
        y = 0
        q = a
        return x, y, q
    ret = Get_ni(b, a % b)
    x = ret[0]
    y = ret[1]
    q = ret[2]
    temp = x
    x = y
    y = temp - a // b * y
    return x, y, q


def result(a_list, m_list):
    for i in range(len(m_list)):
        for j in range(i + 1, len(m_list)):
            if 1 != math.gcd(m_list[i], m_list[j]):
                print("The Chinese remainder theorem cannot be directly utilized.")
                return
    m = 1
    for mi in m_list:
        m *= mi
    Mi_list = Get_Mi(m_list, m)
    Mi_inverse = Get_resMi(Mi_list, m_list)
    x = 0
    for i in range(len(a_list)):
        x += Mi_list[i] * Mi_inverse[i] * a_list[i]
        x %= m
    return x


if __name__ == '__main__':
    print("Please input ai, separated by commas:")
    a_list = list(map(int, input().split(",")))
    print("Please input mi, separated by commas:")
    m_list = list(map(int, input().split(",")))
    print("The final result is：")
    print(result(a_list, m_list))


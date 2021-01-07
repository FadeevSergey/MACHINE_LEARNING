# score - 988/1000
import time
import numpy as np


def get_e(ww, data_x, data_y, i, bb):
    result = 0
    for o in range(len(ww)):
        result += ww[o] * data_y[o] * data_x[i][o]
    return result + bb


def get_rand_not_i(ii, size):
    j = ii
    while j == ii:
        j = np.random.randint(0, size - 1)
    return j


def get_n(x, i, j):
    return 2 * x[i][j] - x[i][i] - x[j][j]


def update_w(ww, y, e_i, e_j, nn, L, H):
    ww = ww - y * (e_i - e_j) / nn
    if ww > H:
        return H
    elif H >= ww > L:
        return ww
    else:
        return L


def update_b(b, i, j, w, x, y, w_old_i, w_old_j, E_i, E_j, c):
    b_1 = b - E_i - y[i] * (w[i] - w_old_i) * x[i][i] - y[j] * (w[j] - w_old_j) * x[i][j]
    b_2 = b - E_j - y[i] * (w[i] - w_old_i) * x[i][j] - y[j] * (w[j] - w_old_j) * x[j][j]

    if c > w[i] > 0:
        return b_1
    elif c > w[j] > c:
        return b_2
    else:
        return (b_1 + b_2) / 2


def svm(x, y, c, start_time):
    w = [0] * len(x)
    bb = 0

    eps = 10 ** (-5.1)
    it = 0
    while time.time() - start_time < 4.500000 or it < 10:
        ch = 0
        for ii in range(len(x)):
            E_i = get_e(w, x, y, ii, bb) - y[ii]

            if (y[ii] * E_i < -(10 ** (-5)) and w[ii] < c) or (y[ii] * E_i > (10 ** (-5)) and w[ii] > 0):
                j = get_rand_not_i(ii, len(x))
                E_j = get_e(w, x, y, j, bb) - y[j]
                w_old_i = w[ii]
                w_old_j = w[j]
                if y[ii] != y[j]:
                    L = max(0, w[j] - w[ii])
                    H = min(c, c + w[j] - w[ii])
                else:
                    L = max(0, w[ii] + w[j] - c)
                    H = min(c, w[ii] + w[j])
                if L == H:
                    continue
                n = get_n(x, ii, j)
                if n >= 0:
                    continue
                w_t = update_w(w[j], y[j], E_i, E_j, n, L, H)
                if abs(w_old_j - w_t) < eps:
                    continue
                w[j] = w_t
                w[ii] = w[ii] + y[ii] * y[j] * (w_old_j - w[j])
                bb = update_b(bb, ii, j, w, x, y, w_old_i, w_old_j, E_i, E_j, c)
                ch += 1
            if ch == 0:
                it += 1
            else:
                it = 0
    return w, bb


def read_dataset(n):
    kernel_values = []
    y = []
    for i in range(n):
        new_line = list(map(int, input().split()))
        kernel_values.append(new_line[:-1])
        y.append(new_line[-1])
    return kernel_values, y


if __name__ == '__main__':
    start_time = time.time()
    n = int(input())
    kernel_values, y = read_dataset(n)
    c = int(input())

    w, b = svm(kernel_values, y, c, start_time)

    for cur_w in w:
        print("%.15f" % cur_w)
    print("%.15f" % b)

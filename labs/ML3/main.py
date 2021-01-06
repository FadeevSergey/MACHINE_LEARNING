import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt


def kernel_linear(x, y):
    return np.dot(x, y)


def kernel_polynomial(x, y, p=3):
    return np.power(1 + np.dot(x, y), p)


def kernel_gauss_radial_basis_func(x, y, bt=5):
    return np.exp(-bt * np.power(np.linalg.norm(x - y), 2))


def get_normalize_dataset(data):
    new_dataset = data.copy()
    for label in data.columns:
        if label != "class":
        #     new_dataset[label] = pandas.factorize(data[label])[0]
        # else:
            min_parameter = data[label].min()
            max_parameter = data[label].max()
            new_dataset[label] = (data[label] - min_parameter) / (max_parameter - min_parameter)
    # print(new_dataset)
    x = new_dataset.values[:, :-1]
    y = new_dataset.values[:, -1]
    # print(x)
    for i in range(len(y)):
        if y[i] == "P":
            y[i] = -1
        else:
            y[i] = 1
    # print(y)

    return x, y


def get_e(ww, data_x, data_y, x, bb, kernel_function):
    result = 0
    for o in range(len(ww)):
        result += ww[o] * data_y[o] * kernel_function(x, data_x[o])
    return result + bb


def get_rand_not_i(ii, size):
    j = ii
    while j == ii:
        j = np.random.randint(0, size - 1)
    return j


def get_n(x_i, x_j, kernel_function):
    return 2 * kernel_function(x_i, x_j) - kernel_function(x_i, x_i) - kernel_function(x_j, x_j)


def update_w(ww, y, e_i, e_j, nn, L, H):
    # print(w, y, E_i, E_j, n, L, H)

    ww = ww - y * (e_i - e_j) / nn
    # print(w)
    # print(w)
    if ww > H:
        return H
    elif H >= ww > L:
        return ww
    else:
        return L


def update_b(b, i, j, w, x, y, w_old_i, w_old_j, E_i, E_j, c, kernel_function):
    b_1 = b - E_i - y[i] * (w[i] - w_old_i) * kernel_function(x[i], x[i]) - y[j] * (w[j] - w_old_j) * kernel_function(x[i], x[j])
    b_2 = b - E_j - y[i] * (w[i] - w_old_i) * kernel_function(x[i], x[j]) - y[j] * (w[j] - w_old_j) * kernel_function(x[j], x[j])

    if c > w[i] > 0:
        return b_1
    elif c > w[j] > c:
        return b_2
    else:
        return (b_1 + b_2) / 2


def predict(w, data_x, data_y, x, b, kernel_function):
    result = 0
    for i in range(len(w)):
        result += w[i] * data_y[i] * kernel_function(x, data_x[i])
    return result + b


def svm(x, y, kernel_function, c):
    w = [0] * len(x)
    bb = 0

    eps = 10 ** (-5)
    it = 0
    while it < 20:
        ch = 0
        for ii in range(len(x)):
            E_i = get_e(w, x, y, x[ii], bb, kernel_function) - y[ii]

            if (y[ii] * E_i < -(10 ** (-8)) and w[ii] < c) or (y[ii] * E_i > (10 ** (-8)) and w[ii] > 0):
                j = get_rand_not_i(ii, len(x))
                E_j = get_e(w, x, y, x[j], bb, kernel_function) - y[j]
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
                n = get_n(x[ii], x[j], kernel_function)
                if n >= 0:
                    continue
                w_t = update_w(w[j], y[j], E_i, E_j, n, L, H)
                if abs(w_old_j - w_t) < eps:
                    continue
                w[j] = w_t
                w[ii] = w[ii] + y[ii] * y[j] * (w_old_j - w[j])
                bb = update_b(bb, ii, j, w, x, y, w_old_i, w_old_j, E_i, E_j, c, kernel_function)
                ch += 1
            if ch == 0:
                it += 1
            else:
                it = 0
    # print(w)
    # print(b)
    return w, bb


P = [2, 3, 4, 5]

BT = [1, 2, 3, 4, 5]

C = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

if __name__ == '__main__':
    # datasets = ["./datasets/chips.csv", "./datasets/geyser.csv"]
    datasets = ["./datasets/chips.csv"]
    kernel_functions = [kernel_linear, kernel_polynomial, kernel_gauss_radial_basis_func]
    for path_to_dataset in datasets:
        score = -1
        for o in (5, 6, 7):
            dataset = pandas.read_csv(path_to_dataset)
            temp_x, temp_y = get_normalize_dataset(dataset)
            # print(temp_x)
            # print(temp_y)
            p = 0
            n = 0
            best_c = -1
            for i in range(len(temp_x)):
                t_x = temp_x[i]
                t_y = temp_y[i]
                new_temp_x = np.delete(temp_x, i, 0)
                new_temp_y = np.delete(temp_y, i, 0)
                print(len(new_temp_y))
                print(len(new_temp_x))
                y_pred = []
                w, b = svm(new_temp_x, new_temp_y, kernel_gauss_radial_basis_func, C[o])
                pr = predict(w, new_temp_x, new_temp_y, t_x, b, kernel_gauss_radial_basis_func)
                pr = -1 if pr <= 0 else 1
                y_pred.append(pr)
                if pr == t_y:
                    p += 1
                else:
                    n += 1

            new_score = p / (p + n)
            if new_score > score:
                score = new_score
                best_c = C[o]
            print(new_score, C[o])
        print(new_score, C[o])

            # w, b = svm(temp_x, temp_y, kernel_linear, C[1])
            # x1_coord = []
            # y1_coord = []
            # x2_coord = []
            # y2_coord = []
            #
            # for i in range(len(temp_x)):
            #     if temp_y[i] == -1:
            #         x1_coord.append(temp_x[i][0])
            #         y1_coord.append(temp_x[i][1])
            #     else:
            #         x2_coord.append(temp_x[i][0])
            #         y2_coord.append(temp_x[i][1])
            #
            # plt.plot(x1_coord, y1_coord, 'ro')
            # plt.plot(x2_coord, y2_coord, 'gs')
            #
            # w, b = svm(temp_x, temp_y, kernel_gauss_radial_basis_func, C[7])
            #
            # ax = plt.gca()
            #
            # xlim = ax.get_xlim()
            # ylim = ax.get_ylim()
            # print(xlim, ylim)
            # xx = np.linspace(xlim[0], xlim[1], 60)
            # yy = np.linspace(ylim[0], ylim[1], 60)
            # YY, XX = np.meshgrid(yy, xx)
            # xy = np.vstack([XX.ravel(), YY.ravel()]).T
            #
            # y_pred = []
            #
            # for obj in xy:
            #     y_pred.append(predict(w, temp_x, temp_y, obj, b, kernel_gauss_radial_basis_func))
            # Z = y_pred
            # Z = np.array(Z).reshape(XX.shape)
            #
            # ax.contourf(XX, YY, Z, levels=[-100, 0, 100], alpha=0.5, colors=['#0000ff', '#ff0000'])
            #
            # ax.contour(XX, YY, Z, levels=[-1, 0, 1], alpha=1, linestyles=['--', '-', '--'], colors='k')
            # C_str = str(C[7])
            # plt.title("chips, kernel - gauss, β = 5, C = " + C_str)
            # plt.show()

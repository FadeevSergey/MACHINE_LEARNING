import numpy as np
import matplotlib.pyplot as plt
import os


class Message:
    def __init__(self, message_class, title, message):
        self.message_class = message_class
        self.title = title
        self.message = message
        self.array_representation = None

    def to_array(self):
        if self.array_representation is None:
            self.array_representation = self.title + self.message
        return self.array_representation


class Bayes:
    def __init__(self, alpha):
        self.alpha = alpha
        self.p = dict()
        self.count_of_classes = dict()
        self.count_of_values = dict()

    def fit(self, X, y):
        for i in range(len(X)):
            if not y[i] in self.count_of_values:
                self.count_of_values[y[i]] = dict()
            if y[i] in self.count_of_classes:
                self.count_of_classes[y[i]] += 1
            else:
                self.count_of_classes[y[i]] = 1

            for j in range(len(X[i])):
                if X[i][j] == 1:
                    if j in self.count_of_values[y[i]]:
                        self.count_of_values[y[i]][j] += 1
                    else:
                        self.count_of_values[y[i]][j] = 1
                else:
                    if j in self.count_of_values[y[i]]:
                        self.count_of_values[y[i]][j] += 0
                    else:
                        self.count_of_values[y[i]][j] = 0

        for i in self.count_of_classes:
            self.p[i] = dict()
            for value in self.count_of_values[i]:
                self.p[i][value] = (self.count_of_values[i][value] + self.alpha) / (
                        self.count_of_classes[i] + self.alpha * 2)

    # what to return == "class" or "probability"

    def predict(self, x, lambdas, what_to_return="class"):
        assert(what_to_return == "class" or what_to_return == "probability")
        best_p = -1
        prediction = -1
        probabilities = dict()

        messages_number = 0
        for i in self.count_of_classes:
            messages_number += self.count_of_classes[i]


        for i in self.count_of_classes:
            cur_p = lambdas[i]
            cur_p += np.log(self.count_of_classes[i] / messages_number)
            for j in range(len(x)):
                if x[j] == 1:
                    cur_p += np.log(self.p[i][j])
                else:
                    cur_p += np.log(1 - self.p[i][j])

            if cur_p > best_p or prediction == -1:
                best_p = cur_p
                prediction = i

            probabilities[i] = cur_p

        # first  - spam
        # second - legit
        # exp(second) / (exp(first) + exp(second))
        # exp(second) / (exp(second) * (exp(first - second) - 1))
        # 1 / (exp(first - second) + 1)

        if what_to_return == "class":
            return prediction
        else:
            return probabilities[0] - probabilities[1]


class NGram:
    def __init__(self, n):
        self.n = n
        self.obj_to_index = dict()
        self.prime_numbers = [1, 23, 37]

    def fit(self, messages, test_messages):
        y = []

        vectors = []
        values = set()
        all_messages = messages + test_messages
        for message in all_messages:
            vector = set()
            y.append(message.message_class)
            title = message.title
            message_text = message.message
            if len(title) != 0:
                vector, values = self.split_array(title, vector, values)

            vector, values = self.split_array(message_text, vector, values)

            vectors.append(vector)

        X = [[0 for _ in range(len(values))] for _ in range(len(vectors))]

        for i, vector in enumerate(vectors):
            for j, value in enumerate(values):
                self.obj_to_index[value] = i
                X[i][j] = 1 if value in vector else 0
        return X[0: len(messages)], y[0: len(messages)], X[len(messages):], y[len(messages):]

    def split_array(self, array, vector, values):
        for i in range(len(array) - (self.n - 1)):
            value = 0
            for j in range(i, i + self.n):
                value += int(array[i]) * self.prime_numbers[j - i]
            vector.add(value)
            values.add(value)

        return vector, values


def process_files(files):
    messages = []
    for path_to_file in files:
        assert (path_to_file.__contains__("spmsg") or path_to_file.__contains__("legit"))
        file = open(path_to_file, 'r')
        message_class = 0 if path_to_file.__contains__("spmsg") else 1
        title = file.readline().replace("\n", "").removeprefix("Subject: ").split(" ")
        if title != [""]:
            title = list(map(lambda s: int(s), title))
        else:
            title = []

        file.readline()
        message = file.readline().replace("\n", "").split(" ")
        message = list(map(int, message))
        file.close()

        new_message = Message(message_class, title, message)
        messages.append(new_message)

    return messages


def print_ROC():
    number_of_test_part = 10
    train_range = range(1, 11)
    train_files = []
    alpha = 30
    n = 1
    for i in train_range:
        cur_files = os.listdir(path="./messages/part" + str(i))
        cur_files = map(lambda s: "./messages/part" + str(i) + "/" + s, cur_files)
        train_files += cur_files

    test_files = os.listdir(path="./messages/part" + str(number_of_test_part))
    test_files = list(map(lambda s: "./messages/part" + str(number_of_test_part) + "/" + s, test_files))

    train_messages = process_files(train_files)
    test_messages = process_files(test_files)

    n_gram = NGram(n)
    X1, y1, X2, y2 = n_gram.fit(train_messages, test_messages)

    model = Bayes(alpha)
    model.fit(X1, y1)
    probabilities = []

    first_class_count = 0
    second_class_count = 0

    for cls in y2:
        if cls == 0:
            first_class_count += 1
        else:
            second_class_count += 1

    x_step = 1 / first_class_count
    y_step = 1 / second_class_count

    for i in range(len(X2)):
        probabilities.append((model.predict(X2[i], [0, 0], what_to_return="probability"), y2[i]))

    probabilities = sorted(probabilities)

    x_points = [0]
    y_points = [0]

    for (_, cls) in probabilities:
        x_prev = x_points[len(x_points) - 1]
        y_prev = y_points[len(y_points) - 1]
        if cls == 0:
            x_points.append(x_prev + x_step)
            y_points.append(y_prev)
        else:
            x_points.append(x_prev)
            y_points.append(y_prev + y_step)

    plt.scatter(x_points, y_points, s=3)
    plt.plot(x_points, y_points)
    plt.grid(True)
    plt.title("ROC curve. " + "Alpha = " + str(alpha) + ", " + str(n) + "-gram")
    plt.show()


def print_dependency_graph():
    accuracy = []
    steps = []

    optimal_lambda_legit = 180
    lambda_legit = 0
    lambda_spam = 0

    while lambda_legit <= optimal_lambda_legit:
        P = 0
        N = 0
        for j in range(1, 11):
            train_files = []
            for i in range(1, 11):

                if i != j:
                    cur_files = os.listdir(path="./messages/part" + str(i))
                    cur_files = map(lambda s: "./messages/part" + str(i) + "/" + s, cur_files)
                    train_files += cur_files

            test_files = os.listdir(path="./messages/part" + str(j))
            test_files = list(map(lambda s: "./messages/part" + str(j) + "/" + s, test_files))

            train_messages = process_files(train_files)
            test_messages = process_files(test_files)

            n_gram = NGram(2)
            X1, y1, X2, y2 = n_gram.fit(train_messages, test_messages)

            model = Bayes(0.00000001)
            model.fit(X1, y1)
            P = 0
            N = 0
            legit_N = 0
            for i in range(len(X2)):
                if model.predict(X2[i], [lambda_spam, lambda_legit]) == y2[i]:
                    P += 1
                else:
                    if y2[i] == 1:
                        legit_N += 1
                    N += 1
        accuracy.append(P / (P + N))
        steps.append(lambda_legit)
        lambda_legit += 22.5

    plt.title("λ_spam = " + str(lambda_spam) + ", optimal λ_legit = " + str(optimal_lambda_legit))
    plt.plot(steps, accuracy)
    plt.scatter(steps, accuracy, s=5)
    plt.xlabel("lambda legit")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    print_ROC()
    print_dependency_graph()

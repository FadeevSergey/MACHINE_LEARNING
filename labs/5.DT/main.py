import pandas as pd
import numpy as np
import matplotlib.pyplot
from collections import Counter
from sklearn.tree import DecisionTreeClassifier


class Dataset:
    def __init__(self, X_train, y_train, X_test, y_test, number):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.number = number


def get_best_depth(X_train, y_train, X_test, y_test):
    best_depth = 0
    best_score = 0

    for j in range(1, len(X_test[0])):
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=j)
        print(y_train)
        tree = tree.fit(X_train, y_train)

        cur_score = tree.score(X_test, y_test)
        if cur_score > best_score:
            best_score = cur_score
            best_depth = tree.max_depth

    return best_depth, best_score


def path_to_dataset(train_or_test: str, number):
    path_to_dir = "./DT_csv"
    dataset_format = ".csv"

    return path_to_dir + "/" + str('{:02}'.format(number)) + "_" + train_or_test + dataset_format


def process_dataset(data):
    x = data.values[:, :-1]
    y = data.values[:, -1]
    return x, y


def get_subdataset(X, y, mtry, sampsize):
    deleted_columns = set(np.random.randint(0, len(X[0]), len(X[0]) - sampsize))
    rows_numbers_to_leave = set(np.random.randint(0, len(X), mtry))

    new_X = []
    new_y = []

    for i in rows_numbers_to_leave:
        new_X.append(X[i])
        new_y.append(y[i])

    for i in deleted_columns:
        for x in new_X:
            x[i] = 0

    return new_X, new_y


def score_of_forest(predictions, answers):
    predictions = np.array(predictions).transpose()

    P = 0
    N = 0

    for i in range(len(answers)):
        if Counter(predictions[i].flat).most_common(1)[0][0] == answers[i]:
            P += 1
        else:
            N += 1

    return P / (P + N)


def random_forests():
    trees_in_forest = 1000

    train_datasets_scores = []
    test_datasets_scores = []
    datasets_numbers = []

    for i in range(1, 22):
        print("log: random_forests() - start processing dataset number " + str(i))

        path_to_train = path_to_dataset("train", i)
        path_to_test = path_to_dataset("test", i)
        train_dataset = pd.read_csv(path_to_train)
        test_dataset = pd.read_csv(path_to_test)

        X_train, y_train = process_dataset(train_dataset)
        X_test, y_test = process_dataset(test_dataset)

        # mtry = int(np.sqrt(len(X_train)))
        # sampsize = int(np.ceil(.632 * len(X_train[0])))
        mtry = int(len(X_train))
        sampsize = int(np.sqrt(len(X_train[0])))
        cur_train_predictions = []
        cur_test_predictions = []

        for _ in range(trees_in_forest):
            cur_X, cur_y = get_subdataset(X_train, y_train, mtry, sampsize)
            tree = DecisionTreeClassifier()
            tree = tree.fit(cur_X, cur_y)
            cur_train_predictions.append(tree.predict(X_train))
            cur_test_predictions.append(tree.predict(X_test))

        train_datasets_scores.append(score_of_forest(cur_train_predictions, y_train))
        test_datasets_scores.append(score_of_forest(cur_test_predictions, y_test))
        datasets_numbers.append(i)

    return train_datasets_scores, test_datasets_scores


def print_random_forest_scores(datasets_number, trees_scores, forests_trains_scores, forests_test_scores):
    for i in range(len(forests_trains_scores)):
        forests_trains_scores[i] = float("{0:.2f}".format(forests_trains_scores[i]))
        forests_test_scores[i] = float("{0:.2f}".format(forests_test_scores[i]))
        trees_scores[i] = float("{0:.2f}".format(trees_scores[i]))


    x = np.arange(len(datasets_number))
    width = 0.3

    fig, ax = matplotlib.pyplot.subplots()
    rects1 = ax.bar(x - width, forests_trains_scores, width, label="Fores train")
    rects2 = ax.bar(x, forests_test_scores, width, label="Fores test")
    rects3 = ax.bar(x + width, trees_scores, width, label="Tree")


    ax.set_ylabel("Scores")
    ax.set_xlabel("Dataset number")
    ax.set_title("Random forests scores")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_number)
    ax.legend()

    for rects in (rects1, rects2, rects3):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width(), height),
                        xytext=(0, 2),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig.tight_layout()
    matplotlib.pyplot.show()


def print_tree_scores(dataset, best_depth, description):
    test_scores = []
    train_scores = []
    depths = []

    for i in range(1, len(dataset.X_train[0])):
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=i)
        tree = tree.fit(dataset.X_train, dataset.y_train)
        test_scores.append(tree.score(dataset.X_test, dataset.y_test))
        train_scores.append(tree.score(dataset.X_train, dataset.y_train))
        depths.append(i)

    matplotlib.pyplot.title(
        description + ". Dataset number - " + str(dataset.number) + ". Best depth = " + str(best_depth))
    matplotlib.pyplot.xlabel("depth")
    matplotlib.pyplot.ylabel("score")
    matplotlib.pyplot.plot(depths, train_scores, label="train")
    matplotlib.pyplot.plot(depths, test_scores, label="test")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


def decision_trees():
    max_depth = 0
    max_depth_score = 0
    min_depth = 100_000_000_000
    min_depth_score = 0

    dataset_scores = []

    for i in range(1, 22):
        print("log: find_min_max_depth() - start processing dataset number " + str(i))
        path_to_train = path_to_dataset("train", i)
        path_to_test = path_to_dataset("test", i)
        train_dataset = pd.read_csv(path_to_train)
        test_dataset = pd.read_csv(path_to_test)

        X_train, y_train = process_dataset(train_dataset)
        X_test, y_test = process_dataset(test_dataset)

        best_depth, score = get_best_depth(X_train, y_train, X_test, y_test)

        if best_depth >= max_depth:
            max_depth = best_depth
            max_depth_dataset = Dataset(X_train, y_train, X_test, y_test, number=i)
        if best_depth <= min_depth:
            min_depth = best_depth
            min_depth_dataset = Dataset(X_train, y_train, X_test, y_test, number=i)

        dataset_scores.append(score)

    print_tree_scores(min_depth_dataset, min_depth, description="Minimal depth of tree")
    print_tree_scores(max_depth_dataset, max_depth, description="Maximal depth of tree")

    return dataset_scores


if __name__ == '__main__':
    trees_scores = decision_trees()
    forests_train_scores, forests_test_scores = random_forests()
    print_random_forest_scores([i for i in range(1, 22)], trees_scores, forests_train_scores, forests_test_scores)


import numpy
import torch
import torchvision
import torchvision.transforms as transfroms
import matplotlib.pyplot as plt
import os


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5, 1, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def neural_network_accuracy(neural_network, test_loader):
    n = 0
    p = 0
    for X, Y in test_loader:
        n += len(X)
        _, Y_predict = torch.max(neural_network(X).data, 1)
        for i, y_predict in enumerate(Y_predict):
            if y_predict == Y[i]:
                p += 1
    return p / n


def load_dataset(dataset):
    transform = torchvision.transforms.Compose([transfroms.ToTensor(), transfroms.Normalize(0.5, 0.5)])

    train_set = dataset(root='./data', train=True, download=True, transform=transform)
    test_set = dataset(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

    return train_set, test_set, train_loader, test_loader


def train_neural_network(neural_network, train_loader, test_loader):
    log_file = open('result/log.txt', 'w')

    optimizer = torch.optim.Adagrad(neural_network.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = neural_network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                accuracy = neural_network_accuracy(neural_network, test_loader)
                log_text = "Epoch - " + str(epoch + 1) + \
                           ", iteration " + str(i + 1) + \
                           ". Loss - " + str("%.15f" % (running_loss / 200)) + \
                           ". Accuracy - " + str("%.10f" % accuracy + "\n")
                log_file.write(log_text)
                running_loss = 0
    log_file.close()


def print_result(xs, ys, probabilities, classes):

    confusion_matrix = []
    probabilities_matrix = []
    img_matrix = []
    for i in range(len(classes)):
        confusion_matrix.append([])
        probabilities_matrix.append([])
        img_matrix.append([])
        for j in range(len(classes)):
            confusion_matrix[i].append(0)
            probabilities_matrix[i].append(0)
            img_matrix[i].append(numpy.zeros((1, 28, 28)))
    pred_ys = probabilities.argmax(axis=1)
    for x, y, pred_y, prob in zip(xs, ys, pred_ys, probabilities):
        confusion_matrix[pred_y][y] += 1
        if probabilities_matrix[pred_y][y] < prob[pred_y]:
            probabilities_matrix[pred_y][y] = prob[pred_y]
            img_matrix[pred_y][y] = x

    print_images_matrix(img_matrix, classes)
    print_confusion_matrix(confusion_matrix, classes)


def print_images_matrix(images_matrix, classes):
    plt.figure(figsize=(len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.subplot(len(classes), len(classes), i * len(classes) + j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images_matrix[i][j].reshape((28, 28)))
            if j == 0:
                plt.ylabel(classes[i])
            if i == 9:
                plt.xlabel(classes[j])
    plt.savefig('result/images_matrix.png')


def print_confusion_matrix(confusion_matrix, classes):
    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    for i, cur_class in enumerate(classes):
        confusion_matrix[i].insert(0, cur_class)
    ax.table(cellText=confusion_matrix, colLabels=[None] + classes, loc='center')
    fig.tight_layout()

    plt.savefig('result/confusion_matrix.png', dpi=600)
    plt.close()


def start():
    dir_name = "result"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    dataset = torchvision.datasets.FashionMNIST
    dataset_train, dataset_test, dataset_train_loader, dataset_test_loader = load_dataset(dataset)
    net = Net()
    train_neural_network(net, dataset_train_loader, dataset_test_loader)
    xs = dataset_test.data.unsqueeze(1).float()
    ys = dataset_test.targets.numpy()
    probabilities = torch.nn.Softmax(dim=1)(net(xs).detach()).numpy()
    print_result(xs, ys, probabilities, dataset_test.classes)


if __name__ == '__main__':
    start()

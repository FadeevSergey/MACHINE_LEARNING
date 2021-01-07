// score - 1000/1000

#include <iostream>
#include <vector>
#include "iomanip"

int n;
std::vector<std::vector<int>> predicts;
std::vector<int> predictsNumber;
int numberOfPredicts;

int falseNegative(int index) {
    int result = 0;
    for(int i = 0; i < n; i++) {
        if(i != index) {
            result += predicts[index][i];
        }
    }
    return result;
}

int falsePositive(int index) {
    int result = 0;
    for(int i = 0; i < n; i++) {
        if(i != index) {
            result += predicts[i][index];
        }
    }
    return result;
}

double precAndRecall(int TP, int second) {
    double result;
    if(TP + second == 0) {
        result = 0;
    } else {
        result = (double)TP / (TP + second);
    }

    return result;
}

double prec(int index) {
    int TP = predicts[index][index];
    int FP = falsePositive(index);

    return precAndRecall(TP, FP);

}

double recall(int index) {
    int TP = predicts[index][index];
    int FN = falseNegative(index);

    return precAndRecall(TP, FN);
}

double fScore(double prec, double recall) {
    double result;

    if(prec + recall == 0) {
        result = 0;
    } else {
        result = 2 * (prec * recall) / (prec + recall);
    }

    return result;
}

double macroFScore() {
    double allRecall = 0;
    double allPrec = 0;

    for(int i = 0; i < n; i++) {
        allRecall += predictsNumber[i] * recall(i);
        allPrec += predictsNumber[i] * prec(i);
    }

    allPrec /= numberOfPredicts;
    allRecall /= numberOfPredicts;

    return fScore(allPrec, allRecall);
}

double microFScore() {
    double score = 0;

    for(int i = 0; i < n; i++) {
        double curPrec = prec(i);
        double curRecall = recall(i);

        score += predictsNumber[i] * fScore(curPrec, curRecall);
    }

    return score / numberOfPredicts;
}

int main() {
    std::cin >> n;
    predicts = std::vector<std::vector<int>>(n, std::vector<int>(n));
    predictsNumber = std::vector<int>(n, 0);
    numberOfPredicts = 0;

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            int tempNumber;
            std::cin >> tempNumber;
            predicts[i][j] = tempNumber;
            predictsNumber[i] += tempNumber;
            numberOfPredicts += tempNumber;
        }
    }

    std::cout << std::setprecision(9) << macroFScore() << "\n" << std::setprecision(9) << microFScore();
}
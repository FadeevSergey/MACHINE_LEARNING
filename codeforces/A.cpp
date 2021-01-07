// score - 1000/1000

#include <iostream>
#include <unordered_set>
#include <vector>

int main() {
    int n, m, k;
    std::cin >> n >> m >> k;

    std::vector<std::unordered_set<int>> classes(m, std::unordered_set<int>());
    std::vector<std::vector<int>> result(k, std::vector<int>());

    for(int i = 0; i < n; i++) {
        int tempClass;
        std::cin >> tempClass;
        classes[tempClass - 1].insert(i + 1);
    }

    int position = 0;

    for(const auto& setOfClasses: classes) {
        for(auto classObject: setOfClasses) {
            result[position].push_back(classObject);
            position = (position + 1) % k;
        }
    }

    for(const auto& componentResult: result) {
        std::cout << componentResult.size() << " ";
        for(auto objectNumber: componentResult) {
            std::cout << objectNumber << " ";
        }
        std::cout << "\n";
    }
}
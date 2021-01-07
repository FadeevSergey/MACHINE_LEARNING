// score - 1000/1000

#include <iostream>
#include <vector>
#include <unordered_map>
#include <iomanip>

int main() {
    size_t k1, k2;
    std::cin >> k1 >> k2;
    size_t n;
    std::cin >> n;

    std::vector<size_t> x1(n);
    std::vector<size_t> x2(n);
    std::vector<size_t> x1_cnt(k1, 0);
    std::vector<size_t> x2_cnt(k2, 0);

    std::unordered_map<size_t, std::unordered_map<size_t, size_t>> count_of_pairs(k1);
    for(size_t i = 0; i < n; i++) {
        std::cin >> x1[i] >> x2[i];
        x1[i]--; x2[i]--;
        x1_cnt[x1[i]]++;
        x2_cnt[x2[i]]++;

        count_of_pairs[x1[i]][x2[i]]++;
    }

    double result = n;

    for(auto const &i: count_of_pairs) {
        for(auto j: i.second) {
            double e = static_cast<double>(x1_cnt[i.first]) * static_cast<double>(x2_cnt[j.first]) / n;
            double o = j.second - e;
            result -= e;
            result += o / e * o;
        }
    }

    std::cout << std::setprecision(15) << result << std::endl;
}
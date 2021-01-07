// score - 1000/1000

#include <iostream>
#include <vector>
#include <unordered_map>
#include <iomanip>
#include <cmath>

using map_pair_to_uint_t = std::unordered_map<size_t, std::unordered_map<size_t, size_t>>;

int main() {
    size_t k1, k2;
    std::cin >> k1 >> k2;
    size_t n;
    std::cin >> n;

    std::vector<uint64_t> x1(n), x2(n);
    std::vector<size_t> x1_cnt(k1, 0), x2_cnt(k2, 0);

    map_pair_to_uint_t count_of_pairs(k1);

    for(size_t i = 0; i < n; i++) {
        std::cin >> x1[i] >> x2[i];
        --x1[i]; --x2[i];
        x1_cnt[x1[i]]++;
        x2_cnt[x2[i]]++;

        count_of_pairs[x1[i]][x2[i]]++;
    }

    double result = 0;
    for(auto const &key_1: count_of_pairs) {
        for(auto const &key_2: key_1.second) {
            auto count_of_pair = static_cast<double>(count_of_pairs[key_1.first][key_2.first]);
            double cur_prob = count_of_pair / x1_cnt[key_1.first];
            double cur_h = cur_prob * (-log(cur_prob));
            result += static_cast<double>(x1_cnt[key_1.first]) / n * cur_h;
        }
    }

    std::cout << std::setprecision(16) << result << std::endl;
}
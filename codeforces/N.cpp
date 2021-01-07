// score - 1000/1000

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

// return pairwise sum of  differences
int64_t pairwise_sum_of_dif(std::vector<int64_t> &vctr) {
    std::sort(vctr.begin(), vctr.end());

    std::vector<int64_t> suf(vctr.size());
    suf[vctr.size() - 1] = vctr[vctr.size() - 1];
    for(int i = vctr.size() - 2; i >= 0; i--) {
        suf[i] = vctr[i] + suf[i + 1];
    }

    uint64_t res = 0;

    for(size_t i = 0; i < vctr.size() - 1; i++) {
        res += suf[i + 1] - (vctr.size() - i - 1) * vctr[i];
    }

    return 2 * res;
}

// return in-class distance 
int64_t in_class_dist(std::unordered_map<int64_t, std::vector<int64_t>> & objects) {
    int64_t result = 0;
    for(auto & cur_class: objects) {
        result += pairwise_sum_of_dif(cur_class.second);
    }
    return result;
}

int main() {
    size_t k, n;
    std::cin >> k >> n;
    std::vector<int64_t> x(n);
    std::vector<int64_t> y(n);

    for(size_t i = 0; i < n; i++) {
        std::cin >> x[i] >> y[i];
    }

    std::unordered_map<int64_t, std::vector<int64_t>> objects;
    for(size_t i = 0; i < x.size(); i++) {
        objects[y[i]].emplace_back(x[i]);
    }

    auto sum = pairwise_sum_of_dif(x);
    auto in_dist = in_class_dist(objects);
    auto out_dist = sum - in_dist;

    std::cout << in_dist << std::endl;
    std::cout << out_dist << std::endl;
}
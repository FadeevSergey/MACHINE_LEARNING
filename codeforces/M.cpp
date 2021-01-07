// score - 1000/1000

#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <algorithm>

template <typename T>

inline T sqr(T number) {
    return number * number;
}

std::vector<int64_t> get_rgs(std::vector<int64_t> const &vctr) {
    std::vector<std::pair<int64_t, size_t>> objects_with_indexes(vctr.size());

    for(size_t i = 0; i < vctr.size(); i++) {
        objects_with_indexes[i] = {vctr[i], i};
    }

    std::sort(objects_with_indexes.begin(), objects_with_indexes.end());
    std::vector<int64_t> result(vctr.size());

    for(size_t i = 0; i < vctr.size(); i++) {
        result[objects_with_indexes[i].second] = i;

    }

    return result;
}

inline std::vector<int64_t> get_sqrt_of_dif_rgs(std::vector<int64_t> const &x_rgs, std::vector<int64_t> const &y_rgs) {
    std::vector<int64_t> result(x_rgs.size());

    for(size_t i = 0; i < x_rgs.size(); i++) {

        result[i] = sqr(y_rgs[i] - x_rgs[i]);
    }

    return result;
}

double spirman(std::vector<int64_t> const &x, std::vector<int64_t> const &y) {
    assert(x.size() == y.size());

    if(x.size() == 1) {
        return 0;
    }

    std::vector<int64_t> x_rgs = get_rgs(x);
    std::vector<int64_t> y_rgs = get_rgs(y);

    auto sqrt_of_dif_rgs = get_sqrt_of_dif_rgs(x_rgs, y_rgs);
    double result = 1;
    for(auto const & cur_dif: sqrt_of_dif_rgs) {
        result -= (6.0 * cur_dif) / (x.size() * (x.size() - 1.0) * (x.size() + 1.0));
    }

    return result;
}

int main() {
    size_t n;
    std::cin >> n;
    std::vector<int64_t> x(n), y(n);
    for(size_t i = 0; i < n; i++) {
        std::cin >> x[i] >> y[i];
    }

    std::cout << std::setprecision(9) << spirman(x, y) << std::endl;
}

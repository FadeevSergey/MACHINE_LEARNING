// score - 1000/1000

#include <iostream>
#include <vector>
#include <iomanip>

double get_m_of_sqr(std::vector<int64_t> const& vctr) {
    double result = 0;
    double size_double = vctr.size();
    for(size_t i = 0; i < vctr.size(); i++) {
        result += (double)vctr[i] / size_double * vctr[i];
    }

    return result;
}

std::vector<double> get_prob(std::vector<int64_t> const& vctr, size_t k) {
    std::vector<double> x_prob(k, 0);
    for(size_t i = 0; i < vctr.size(); i++) {
        x_prob[vctr[i] - 1] += 1 / (double)vctr.size();
    }
    return x_prob;
}

std::vector<double> get_m(std::vector<int64_t> const& x, std::vector<int64_t> const& y, size_t k) {
    std::vector<double> y_m(k, 0);
    for(size_t i = 0; i < x.size(); i++) {
        y_m[x[i] - 1] += (double)y[i] / x.size();
    }
    return y_m;
}

int main() {
    size_t k, n;

    std::cin >> k >> n;
    std::vector<int64_t> x(n);
    std::vector<int64_t> y(n);

    for(size_t i = 0; i < n; i++) {
        std::cin >> x[i] >> y[i];
    }

    auto m_of_sqr = get_m_of_sqr(y);
    auto y_m = get_m(x, y, k);
    auto x_prob = get_prob(x, k);

    double result = m_of_sqr;

    for(size_t i = 0; i < k; i++) {
        result -= x_prob[i] == 0 ? 0 : (y_m[i] / x_prob[i] * y_m[i]);
    }

    std::cout << std::setprecision(9) << result << std::endl;
}
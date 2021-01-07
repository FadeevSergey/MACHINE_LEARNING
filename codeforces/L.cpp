// score - 1000/1000

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iomanip>

inline double sqr(double num) {
    return num * num;
}

double average(std::vector<int64_t> const &vctr) {
    int64_t sum = 0;
    for(auto const & value: vctr) {
        sum += value;
    }

    return (double)sum / (vctr.size());
}

double dispers_sqr(std::vector<int64_t> const &vctr) {
    double result = 0;

    double vctr_average = average(vctr);

    for(auto const & value: vctr) {
        result += sqr(value - vctr_average);
    }
    return result;
}


double cov(std::vector<int64_t> const &x, std::vector<int64_t> const &y) {
    double av_x = average(x);
    double av_y = average(y);

    assert(x.size() == y.size());

    double result = 0;

    for(int i = 0; i < x.size(); i++) {
        result += (x[i] - av_x) * (y[i] - av_y);
    }

    return result;
}

double pirson(std::vector<int64_t> const &x, std::vector<int64_t> const &y) {
    double dispersion_x = dispers_sqr(x);
    double dispersion_y = dispers_sqr(y);

    if(dispersion_x == 0 || dispersion_y == 0) {
        return 0;
    } else {
        return cov(x, y) / sqrt(dispersion_x * dispersion_y);
    }
}

int main() {
    size_t n;
    std::cin >> n;
    std::vector<int64_t> x(n), y(n);
    for(size_t i = 0; i < n; i++) {
        std::cin >> x[i] >> y[i];
    }

    std::cout << std::setprecision(10) << pirson(x, y) << std::endl;
}
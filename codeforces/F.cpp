// score - 1000/1000

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>

inline long double accumulate(std::vector<long double> const& vector) {
    long double result = 0;
    for(long double num: vector) {
        result += num;
    }
    return result;
}

void normalize_vector(std::vector<long double> &vector) {
    long double sum = accumulate(vector);
    for(auto &num: vector) {
        num /= sum;
    }
}

int main() {
    size_t k;
    std::cin >> k;
    std::vector<long double> lambdas(k);
    for(size_t i = 0; i < k; i++) {
        std::cin >> lambdas[i];
    }
    size_t alpha;
    std::cin >> alpha;
    size_t n;
    std::cin >> n;

    std::vector<size_t> count_of_classes(k, 0);
    std::unordered_map<size_t, std::vector<std::unordered_set<std::string>>> messages(k);
    std::unordered_set<std::string> words;
    for(size_t i = 0; i < n; i++) {
        size_t cur_class;
        size_t count_of_words;
        std::cin >> cur_class >> count_of_words;

        count_of_classes[cur_class - 1]++;
        messages[cur_class - 1].emplace_back(*(new std::unordered_set<std::string>()));
        for(size_t j = 0; j < count_of_words; j++) {
            std::string cur_word;
            std::cin >> cur_word;
            words.insert(cur_word);
            messages[cur_class - 1].back().insert(cur_word);
        }
    }

    std::unordered_map<size_t, std::unordered_map<std::string, long double>> words_prob(k);

    for(size_t i = 0; i < k; i++) {
        for(auto const &cur_word: words) {
            size_t count_classes_with_word = 0;
            for(const auto& cur_message: messages[i]) {
                if(cur_message.find(cur_word) != cur_message.end())
                    count_classes_with_word++;
            }
            auto num =  static_cast<long double>(count_classes_with_word + static_cast<size_t>(alpha));
            auto denom = static_cast<long double>(count_of_classes[i] + static_cast<size_t>(alpha) * 2);
            words_prob[i][cur_word] = num / denom;
        }
    }

    size_t m;
    std::cin >> m;
    for(size_t i = 0; i < m; i++) {
        size_t count_of_words;
        std::cin >> count_of_words;
        std::unordered_set<std::string> new_message(count_of_words);
        for(size_t j = 0; j < count_of_words; j++) {
            std::string cur_word;
            std::cin >> cur_word;
            new_message.insert(cur_word);
        }

        std::vector<long double> new_answer(k);

        for(size_t cur_class = 0; cur_class < k; cur_class++) {
            long double num = lambdas[cur_class] * static_cast<long double>(count_of_classes[cur_class]) / n;
            long double denom = 0;

            for(auto const &cur_prob: words_prob[cur_class]) {
                if(new_message.find(cur_prob.first) == new_message.end()) {
                    num *= (1 - cur_prob.second);
                } else {
                    num *= cur_prob.second;
                }
            }

            for(size_t temp_class = 0; temp_class < k; temp_class++) {
                long double prob_word_in_class = static_cast<long double>(count_of_classes[temp_class]) / n;
                for(auto const &cur_prob: words_prob[temp_class]) {
                    if(new_message.find(cur_prob.first) == new_message.end()) {
                        prob_word_in_class *= (1 - cur_prob.second);
                    } else {
                        prob_word_in_class *= cur_prob.second;
                    }
                }
                denom += prob_word_in_class;
            }

            new_answer[cur_class] = num / denom;
        }
        normalize_vector(new_answer);
        for(long double p: new_answer) {
            std::cout << std::setprecision(15) << p << " ";
        }
        std::cout << std::endl;
    }
}

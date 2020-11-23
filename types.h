//
// Created by house on 18-6-1.
//

#ifndef ELIS2_0_TYPES_H
#define ELIS2_0_TYPES_H

#include <cmath>
#include <vector>
#include <set>
#include <algorithm>

using namespace std;

struct mydata {
    vector<vector<double>> data;
    vector<int> label;
    size_t num, len;
    size_t classes;
};

struct shapelet {
    vector<vector<double>> shape;
    vector<int> len;
};

struct regression {
    vector<double> w;
    double w0;
};

struct ELIS {
    size_t classes;
    size_t *numbers;
    shapelet *shapelets;
    regression *regressions;
};

struct parameters {
    double regular;
    int epoch;
    double rate;

    parameters(double reg, int epo, double rat) {
        regular = reg;
        epoch = epo;
        rate = rat;
    }
};

struct statistics {
    double max_value, min_value;
    double bucket;
    size_t min_len;
    vector<size_t> datacnt;
};

struct paaword {
    double score;
    size_t window;
    vector<int> vec;
    set<size_t> cov;

    paaword() {
        score = 0;
        window = 0;
        vec.clear();
        cov.clear();
    }

    bool operator<(const paaword f) const {
        if (fabs(score - f.score) > 1e-8) {
            return score > f.score;
        }
        if (cov.size() != f.cov.size()) {
            return cov.size() > f.cov.size();
        }
        return window * vec.size() < f.window * f.vec.size();
    }

    bool operator==(const paaword f) const {
        vector<int> a, b;
        a.clear();
        b.clear();

        for (int i : vec) {
            for (size_t j = 0; j < window; ++j) {
                a.push_back(i);
            }
        }
        a.push_back(100);
        for (int i : f.vec) {
            for (size_t j = 0; j < f.window; ++j) {
                b.push_back(i);
            }
        }
        b.push_back(-100);

        size_t rep = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b.size(); ++j) {
                for (size_t k = 0; i + k < a.size() && j + k < b.size(); ++k) {
                    if (abs(a[i + k] - b[j + k]) > 1) {
                        rep = max(rep, k);
                        break;
                    }
                }
            }
        }

        return rep * 10 > max(a.size(), b.size()) * 9;
    }
};

#endif //ELIS2_0_TYPES_H

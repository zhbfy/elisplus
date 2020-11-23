//
// Created by house on 18-6-7.
//

//#include <Eigen/Dense>
#include <map>
#include <ctime>
#include <random>
#include "../../tool software/eigen/eigen/Eigen/Dense"
#include "types.h"
#include "utils.h"

using namespace std;

#define D_NUM 6
//#define D_NUM 1
#define PAA_BUCKET 10
#define MIN_LENGTH 8
#define EPS 1e-8
#define FUZZY 1
#define MAX_CLASS 100

const int PAA_DIMENSION[D_NUM] = {2, 4, 8, 16, 32, 64};
//const int PAA_DIMENSION[D_NUM] = {4};

mydata x;
ELIS elis;
statistics statis;
map<vector<int>, set<size_t>> tree[MAX_CLASS];
vector<paaword> candidates[MAX_CLASS];

void Statistic() {
    for (int i = 0; i < x.classes; ++i) {
        statis.datacnt.push_back(0);
    }
    statis.min_value = Eigen::Map<Eigen::VectorXd>(x.data[0].data(), x.len).minCoeff();
    statis.max_value = Eigen::Map<Eigen::VectorXd>(x.data[0].data(), x.len).maxCoeff();
    for (int i = 0; i < x.num; ++i) {
        ++statis.datacnt[x.label[i]];
        statis.min_value = min(statis.min_value, Eigen::Map<Eigen::VectorXd>(x.data[i].data(), x.len).minCoeff());
        statis.max_value = max(statis.max_value, Eigen::Map<Eigen::VectorXd>(x.data[i].data(), x.len).maxCoeff());
    }

    statis.bucket = (statis.max_value - statis.min_value) / PAA_BUCKET;
    statis.bucket += statis.bucket / 100;

    statis.min_len = (x.len / 20 + MIN_LENGTH - 1) / MIN_LENGTH * MIN_LENGTH;
}

double PAA2TS(int x) {
    return statis.min_value + statis.bucket * (0.5 + x);
}

void InsertMap(const vector<int> &qu, size_t offset, size_t d, size_t tsid, size_t len, size_t start) {
    size_t window = len / PAA_DIMENSION[d];
    size_t s_pos = start + offset * window;
    for (size_t i = 0; i < PAA_DIMENSION[d]; ++i) {
        for (size_t j = 0; j < window; ++j) {
            if (fabs(x.data[tsid][s_pos + i * window + j] - PAA2TS(qu[offset + i])) > statis.bucket * FUZZY) {
                return;
            }
        }
    }
    vector<int> f;
    for (int i = 0; i < PAA_DIMENSION[d]; ++i) {
        f.push_back(qu[offset + i]);
    }
    tree[x.label[tsid]][f].insert(tsid);
}

set<size_t> QueryMap(const vector<int> &qu, size_t offset, size_t d, size_t classid) {
    set<size_t> cnt;
    vector<int> f;
    for (int i = 0; i < PAA_DIMENSION[d]; ++i) {
        f.push_back(qu[offset + i]);
    }
    if (tree[classid].count(f)) {
        cnt.insert(tree[classid][f].begin(), tree[classid][f].end());
    }
    for (int i = 0; i < PAA_DIMENSION[d]; ++i) {
        f[i] -= 1;
        if (tree[classid].count(f)) {
            cnt.insert(tree[classid][f].begin(), tree[classid][f].end());
        }
        f[i] += 2;
        if (tree[classid].count(f)) {
            cnt.insert(tree[classid][f].begin(), tree[classid][f].end());
        }
        f[i] -= 1;
    }
    return cnt;
}

void CreatePAAWords(size_t len, size_t d) {
    size_t window = len / PAA_DIMENSION[d];
    for (size_t i = 0; i < x.num; ++i) {
        for (size_t start = 0; start < window; start+=1) {
            vector<int> qu;

            // PAA
            for (size_t offset = start; offset + window <= x.len; offset += window) {
                double avg = 0;
                for (size_t j = offset; j < offset + window; ++j) {
                    avg += x.data[i][j];
                }
                qu.push_back((int) floor((avg / window - statis.min_value) / statis.bucket));
            }

            // insert into hash map
            for (size_t j = 0; j + PAA_DIMENSION[d] <= qu.size(); ++j) {
                InsertMap(qu, j, d, i, len, start);
            }
        }
    }
}

void CalculateTFIDF(size_t len, size_t d) {
    size_t window = len / PAA_DIMENSION[d];
	for (size_t i = 0; i < x.num; ++i) {
		for (size_t start = 0; start < window; start += 1) {
            vector<int> qu;

			double TT1 = clock();
            // PAA
            for (size_t offset = start; offset + window <= x.len; offset += window) {
                double avg = 0;
                for (size_t j = offset; j < offset + window; ++j) {
                    avg += x.data[i][j];
                }
                qu.push_back((int) floor((avg / window - statis.min_value) / statis.bucket));
            }
			double TT2 = clock();
            // query from hash map
            for (size_t j = 0; j + PAA_DIMENSION[d] <= qu.size(); ++j) {
				//printf("i:%d,j:%d,k:%d,d:%d\n", i, j, start,d);
                int inclasses = 0;

                vector<set<size_t>> select;
                for (size_t k = 0; k < x.classes; ++k) {
                    select.push_back(QueryMap(qu, j, d, k));
                    if (!select[k].empty()) {
                        ++inclasses;
                    }
                }

                if (inclasses != 1) {
                    continue;
                }

                for (size_t k = 0; k < x.classes; ++k) {
                    double tf = (double) (select[k].size()) / statis.datacnt[k];
                    if (tf < EPS) {
                        continue;
                    }
                    paaword now;
                    now.window = window;
                    now.score = tf;
                    now.cov = select[k];
                    for (size_t cp = 0; cp < PAA_DIMENSION[d]; ++cp) {
                        now.vec.push_back(qu[j + cp]);
                    }
                    candidates[k].push_back(now);
                }
            }
			double TT3 = clock();
			printf("Use Time 11:%f\n", ((TT2 - TT1) / CLOCKS_PER_SEC));
			printf("Use Time 12:%f\n", ((TT3 - TT2) / CLOCKS_PER_SEC));
        }
    }
}

bool CoverGain(const paaword &now, map<size_t, int> &cover, int enough_cover) {
    for (size_t cov : now.cov) {
        if (cover[cov] < enough_cover) {
            return true;
        }
    }
    return false;
}

bool Similar(const paaword &now, vector<paaword> &tmp_result, map<size_t, int> &cover, int enough_cover,
             int &total_cover) {
    for (paaword &item : tmp_result) {
        if (now == item) {
            for (size_t cov : now.cov) {
                if (!item.cov.count(cov)) {
                    item.cov.insert(cov);
                    if (cover[cov] < enough_cover) {
                        ++cover[cov];
                        ++total_cover;
                    }
                }
            }
            return true;
        }
    }
    return false;
}

double CalculateDis(const paaword &paa, size_t tsid) {
    double dis = -1;
    size_t len = paa.window * paa.vec.size();
    for (size_t s_pos = 0; s_pos + len <= x.len; ++s_pos) {
        double distance = 0;
        for (size_t i = 0; i < paa.vec.size(); ++i) {
            double tmpv = PAA2TS(paa.vec[i]);
            for (size_t j = 0; j < paa.window; ++j) {
                double tmp_dis = tmpv - x.data[tsid][s_pos + i * paa.window + j];
                distance += tmp_dis * tmp_dis;
            }
        }
        distance /= len;
        if (dis < 0 || dis > distance) {
            dis = distance;
        }
    }
    return dis;
}

struct disvec {
    double dis;
    size_t idx;

    bool operator<(const disvec f) const {
        return dis < f.dis;
    }
};

double GetEntropy(double a, double b) {
    double ans = 0;
    if (a > EPS) {
        double p = a / (a + b);
        ans -= p * log(p);
    }
    if (b > EPS) {
        double q = b / (a + b);
        ans -= q * log(q);
    }
    return ans;
}

double Check(size_t c, const vector<paaword> &tmp_result) {
    auto dis = new disvec[x.num];
    auto tmp = new disvec[x.num];
    for (size_t i = 0; i < x.num; ++i) {
        dis[i].dis = 0;
        dis[i].idx = i;
    }
    for (const paaword &paa : tmp_result) {
        for (size_t i = 0; i < x.num; ++i) {
            tmp[i].dis = CalculateDis(paa, i);
            tmp[i].idx = i;
        }
        sort(tmp, tmp + x.num);
        int numc = 0;
        size_t best_pos = 0;
        double best_entropy = GetEntropy(statis.datacnt[c], x.num - statis.datacnt[c]);
        for (size_t i = 1; i < x.num; ++i) {
            if (x.label[tmp[i - 1].idx] == c) {
                ++numc;
            }
            double tmp_entropy = GetEntropy(numc, i - numc) +
                                 GetEntropy(statis.datacnt[c] - numc, (x.num - i) - (statis.datacnt[c] - numc));
            if (tmp_entropy < best_entropy) {
                best_entropy = tmp_entropy;
                best_pos = i;
            }
        }
        for (size_t i = 0; i < x.num; ++i) {
            dis[tmp[i].idx].dis += tmp[i].dis - tmp[best_pos].dis;
        }
    }
    int numc = 0;
    sort(dis, dis + x.num);
    double entropy = GetEntropy(statis.datacnt[c], x.num - statis.datacnt[c]);
    for (size_t i = 1; i < x.num; ++i) {
        if (x.label[dis[i].idx] == c) {
            ++numc;
        }
        double tmp_entropy = GetEntropy(numc, i - numc) +
                             GetEntropy(statis.datacnt[c] - numc, (x.num - i) - (statis.datacnt[c] - numc));
        if (tmp_entropy < entropy) {
            entropy = tmp_entropy;
        }
    }
    delete[] dis;
    delete[] tmp;
    return entropy;
}

void DiscoverShapelets() {
	double t1 = clock();
    for (size_t len = statis.min_len; len <= x.len/2; len += statis.min_len) {
        for (size_t d = 0; d < D_NUM; ++d) {
            if (len % PAA_DIMENSION[d] != 0 || len == PAA_DIMENSION[d]) {
                continue;
            }
            for (size_t i = 0; i < x.classes; ++i) {
                tree[i].clear();
            }
			double T1 = clock();
            CreatePAAWords(len, d);
			double T2 = clock();
            CalculateTFIDF(len, d);
			double T3 = clock();
			printf("Use Time:%f\n", ((T2-T1) / CLOCKS_PER_SEC));
			printf("Use Time:%f\n", ((T3-T2) / CLOCKS_PER_SEC));
        }
    }
	double t2 = clock();
    for (size_t c = 0; c < x.classes; ++c) {
        sort(candidates[c].begin(), candidates[c].end());

        int ffff = 0;
        double best_wrong = 1e100;
        vector<paaword> result;
        for (int enough_cover = 1; enough_cover <= 5; ++enough_cover) {
            vector<paaword> tmp_result;
            tmp_result.clear();
            map<size_t, int> cover;
            cover.clear();
            int total_cover = 0;

            for (size_t now_id = 0; now_id < candidates[c].size()
                                    && total_cover < statis.datacnt[c] * enough_cover; ++now_id) {
                paaword now = candidates[c][now_id];
                if (CoverGain(now, cover, enough_cover) &&
                    !Similar(now, tmp_result, cover, enough_cover, total_cover)) {
                    tmp_result.push_back(now);
                    for (size_t cov : now.cov) {
                        if (cover[cov] < enough_cover) {
                            ++cover[cov];
                            ++total_cover;
                        }
                    }
                }
            }

            double tmp_wrong = Check(c, tmp_result);
            if (tmp_wrong < best_wrong) {
                best_wrong = tmp_wrong;
                result = tmp_result;
                ffff = enough_cover;
                if (best_wrong == 0) {
                    break;
                }
            }
            printf("class %d cover %d entropy %f\n", c, enough_cover, tmp_wrong);
        }
        printf("best cover %d\n", ffff);

        elis.numbers[c] = result.size();
        for (paaword &now : result) {
            elis.shapelets[c].len.push_back((int) (now.window * now.vec.size()));
            vector<double> shape;
            shape.clear();
            for (int value: now.vec) {
                double v = PAA2TS(value);
                for (int i = 0; i < now.window; ++i) {
                    shape.push_back(v);
                }
            }
            elis.shapelets[c].shape.push_back(shape);
            elis.regressions[c].w.push_back(-(double) now.cov.size() / statis.datacnt[c]);
        }
    }
	double t3 = clock();
	double dur1 = (double)(t2 - t1);
	double dur2 = (double)(t3 - t2);
	printf("Use Time:%f\n", (dur1 / CLOCKS_PER_SEC));
	printf("Use Time:%f\n", (dur2 / CLOCKS_PER_SEC));
}

void GenData(int rep) {
    size_t tmpn = x.num;
    random_device rd{};
    mt19937 gen1{rd()};
    uniform_int_distribution<> uni(1, x.len - 1);
    default_random_engine gen2;
    normal_distribution<double> nor(0, 0.1);
    x.num *= rep + 1;
    while (rep--) {
        for (int i = 0; i < tmpn; ++i) {
            x.label.push_back(x.label[i]);
            vector<double> tmp;
            tmp.clear();
            int id = uni(gen1);
            tmp.push_back(x.data[i][id]);
            for (size_t j = (id + 1) % x.len; j != id; j = (j + 1) % x.len) {
                tmp.push_back(x.data[i][j]);
            }
            for (int j = 0; j < x.len; ++j) {
                tmp[j] += nor(gen2);
            }
            x.data.push_back(tmp);
        }
    }
}

int main(int argc, char *argv[]) {

    double start_time = clock();

    printf("%s\n", argv[1]);
    string name = argv[1];
    ReadData(name, x);

    if (argc == 3) {
        int rep = stoi(argv[2]);
        GenData(rep);
    }

    elis.classes = x.classes;
    elis.numbers = new size_t[elis.classes];
    elis.shapelets = new shapelet[elis.classes];
    elis.regressions = new regression[elis.classes];

    Statistic();

    DiscoverShapelets();

    WriteModel_dis(elis, true);

    delete[] elis.numbers;
    delete[] elis.shapelets;
    delete[] elis.regressions;

    double end_time = clock();
    printf("Total Running Time = %.3f sec\n", (end_time - start_time) / CLOCKS_PER_SEC);

    return 0;
}

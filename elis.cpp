//
// Created by house on 18-6-2.
//

//#include "types.h"
#include "elis.h"
//#include "stdfxa.h"

#define alpha -25

void ConvertX(const mydata &x, Eigen::MatrixXd &f, Eigen::MatrixXd &d) {
    for (int i = 0; i < x.num; ++i) {
        for (int j = 0; j < x.len; ++j) {
            f(i, j) = x.data[i][j];
        }
        for (int j = 0; j < x.classes; ++j) {
            d(i, j) = (x.label[i] == j) ? 1 : 0;
        }
    }
}

void GradientDescent(const Eigen::MatrixXd &ts, const Eigen::MatrixXd &y, const parameters &p, const size_t all_data,
                     ELIS &elis) {
    vector<Eigen::MatrixXd> save_dis;
    vector<Eigen::MatrixXd> save_e_dis;
    vector<Eigen::MatrixXd> save_sum_e_dis;
    vector<vector<Eigen::MatrixXd> > save_sub;
	
	double aa = 0;

    for (int c = 0; c < elis.classes; ++c) {
        Eigen::MatrixXd m(ts.rows(), elis.numbers[c]);
        save_dis.clear();
        save_e_dis.clear();
        save_sum_e_dis.clear();
        save_sub.clear();

		

        for (int i = 0; i < elis.numbers[c]; ++i) {
            int length = elis.shapelets[c].len[i];
            vector<Eigen::MatrixXd> tmp_save_sub;
            tmp_save_sub.clear();

            Eigen::Map<Eigen::VectorXd> shape(elis.shapelets[c].shape[i].data(), length);

            Eigen::MatrixXd dis(ts.rows(), ts.cols() - length + 1);
            for (int j = 0; j <= ts.cols() - length; ++j) {
                Eigen::MatrixXd sub = ts.middleCols(j, length).rowwise() - shape.transpose();
                dis.col(j) = sub.rowwise().squaredNorm() / length;

                tmp_save_sub.emplace_back(sub);
            }

            Eigen::MatrixXd e_dis = (dis * alpha).array().exp();
            Eigen::MatrixXd sum_e_dis = e_dis.rowwise().sum().array() + 1e-20;
            m.col(i) = dis.cwiseProduct(e_dis).rowwise().sum().cwiseQuotient(sum_e_dis);

            save_dis.push_back(dis);
            save_e_dis.push_back(e_dis);
            save_sum_e_dis.push_back(sum_e_dis);
            save_sub.push_back(tmp_save_sub);
        }

        Eigen::Map<Eigen::VectorXd> w(elis.regressions[c].w.data(), elis.numbers[c]);
        Eigen::VectorXd predict = 1.0 / ((-((m * w).array() + elis.regressions[c].w0)).array().exp() + 1);

        Eigen::MatrixXd v = y.col(c) - predict;

		//cout << " | error: " << v.sum()  << endl;

		//aa += v.sum();

        elis.regressions[c].w0 += v.sum() / ts.rows() * p.rate;

//        Eigen::VectorXd save_w = w + ((m.transpose() * v - w * 2 * p.regular) / ts.rows() * p.rate);
        Eigen::VectorXd save_w = w + ((m.transpose() * v) / ts.rows() - (w * 2 * p.regular) / all_data) * p.rate;

        for (int i = 0; i < elis.numbers[c]; ++i) {
            int length = elis.shapelets[c].len[i];

            Eigen::Map<Eigen::VectorXd> shape(elis.shapelets[c].shape[i].data(), length);

            Eigen::MatrixXd dis = save_dis[i];
            Eigen::MatrixXd e_dis = save_e_dis[i];
            Eigen::MatrixXd sum_e_dis = save_sum_e_dis[i];

            for (int j = 0; j <= ts.cols() - length; ++j) {
                Eigen::MatrixXd tmp =
                        ((dis.col(j) - m.col(i)) * alpha + Eigen::MatrixXd::Constant(ts.rows(), 1, 1)).cwiseProduct(
                                e_dis.col(j)).cwiseQuotient(sum_e_dis) * 2 / length;
                Eigen::MatrixXd sub = save_sub[i][j].transpose() * tmp.asDiagonal();
                shape += (-sub * v * elis.regressions[c].w[i]) / ts.rows() * p.rate;
//                shape += ((shape - ts.middleCols(j, length).transpose()) * tmp.asDiagonal() * v * elis.regressions[c].w[i]) / ts.rows() * p.rate;
            }
        }

        w = save_w;
    }
	//cout << " | error: " << aa << endl;
}

void Train(const mydata &x, parameters &p, ELIS &elis) {
    Eigen::MatrixXd ts(x.num, x.len);
    Eigen::MatrixXd y(x.num, x.classes);
    ConvertX(x, ts, y);

    for (int epoch = 0; epoch < p.epoch; ++epoch) {
        if (epoch % 100 == 0) {
            if (epoch != 0) {
                p.rate *= 0.99;
            }
        }
        for (int i = 0; i < x.num; ++i) {
            GradientDescent(ts.row(i), y.row(i), p, x.num, elis);
        }
    }
}

int Test(const mydata &x, const ELIS &elis) {
    Eigen::MatrixXd ts(x.num, x.len);
    Eigen::MatrixXd y(x.num, x.classes);
    ConvertX(x, ts, y);

    Eigen::MatrixXd res(x.num, x.classes);

    for (int c = 0; c < elis.classes; ++c) {
        Eigen::MatrixXd m(ts.rows(), elis.numbers[c]);
        for (int i = 0; i < elis.numbers[c]; ++i) {
            int length = elis.shapelets[c].len[i];

            Eigen::Map<Eigen::VectorXd> shape(elis.shapelets[c].shape[i].data(), length);

            Eigen::MatrixXd dis(ts.rows(), ts.cols() - length + 1);
            for (int j = 0; j <= ts.cols() - length; ++j) {
                dis.col(j) = (ts.middleCols(j, length).rowwise() - shape.transpose()).rowwise().squaredNorm() / length;
            }

            Eigen::MatrixXd e_dis = (dis * alpha).array().exp();
            m.col(i) = dis.cwiseProduct(e_dis).rowwise().sum().cwiseQuotient(e_dis.rowwise().sum());
        }

        Eigen::Map<Eigen::VectorXd> w(elis.regressions[c].w.data(), elis.numbers[c]);
        res.col(c) = 1.0 / ((-((m * w).array() + elis.regressions[c].w0)).array().exp() + 1);
    }

    Eigen::MatrixXd ans = res.rowwise().maxCoeff();

    int accuracy = 0;
    for (int i = 0; i < x.num; ++i) {
        if (res(i, x.label[i]) == ans(i, 0)) {
            ++accuracy;
        }
    }

    cout << accuracy << " + " << x.num - accuracy;
    return accuracy;
}

void Train(const mydata &x, const mydata &f, parameters &p, ELIS &elis) {
    Eigen::MatrixXd ts(x.num, x.len);
    Eigen::MatrixXd y(x.num, x.classes);
    ConvertX(x, ts, y);

    int best = 0;
    for (int epoch = 0; epoch < p.epoch; ++epoch) {
        if (epoch % 1 == 0) {
            cout << "epoch " << epoch << ": ";
//            Test(x, elis);
//            cout << " | ";
            int acc = Test(f, elis);
            cout << endl;
            if (best < acc) {
                best = acc;
            }
            if (epoch != 0) {
                p.rate *= 0.99;
//				p.rate *= ((p.epoch - epoch -1 )/(p.epoch-epoch));
            }
        }
        for (int i = 0; i < x.num; ++i) {
            GradientDescent(ts.row(i), y.row(i), p, x.num, elis);
        }
    }
    cout << "final state: ";
    int acc = Test(f, elis);
    if (best < acc) {
        best = acc;
    }
	//double rate = 0;
    cout << " | best accuracy: " << best << " + " << f.num - best << endl;
	
}

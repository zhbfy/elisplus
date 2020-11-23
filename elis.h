//
// Created by house on 18-6-4.
//
//#pragma once

#ifndef ELIS2_0_ELIS_H
#define ELIS2_0_ELIS_H

//#include <Eigen/Dense>
#include <iostream>
#include "types.h"
#include "../../tool software/eigen/eigen/Eigen/Dense"

void ConvertX(const mydata &x, Eigen::MatrixXd &f, Eigen::MatrixXd &d);

void GradientDescent(const Eigen::MatrixXd &ts, const Eigen::MatrixXd &y, const parameters &p, const size_t all_data, ELIS &elis,double &error);
//void GradientDescent(const Eigen::MatrixXd &ts, const Eigen::MatrixXd &y, const parameters &p, const size_t all_data, ELIS &elis, int epoch, int maxepoch, Eigen::MatrixXd m_t, Eigen::MatrixXd v_t);
void Train(const mydata &x, parameters &p, ELIS &elis);
int Test(const mydata &x, const ELIS &elis);
void Train(const mydata &x, const mydata &f, parameters &p, ELIS &elis);

#endif //ELIS2_0_ELIS_H

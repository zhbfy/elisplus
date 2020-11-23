//
// Created by house on 18-6-4.
//

#ifndef ELIS2_0_UTILS_H
#define ELIS2_0_UTILS_H

#include <string>
#include <fstream>
#include <set>

#include "types.h"

void SplitStringToDoubleArray(const string &s, const string &c, vector<double> &v);
void ReadData(const string &file_name, mydata &f);
void App_data(mydata &f, mydata &q, int window);
void App_data_smooth(mydata &f, mydata &q, int window);
void ReadELIS(const bool training, ELIS &elis);
void App_shapelet(ELIS &elis1, ELIS &elis, int window);
void App_shapelet_smooth(ELIS &elis1, ELIS &elis, int window);
void WriteModel(const ELIS &elis, const bool discovering);
void WriteModel_dis(const ELIS &elis, const bool discovering);
#endif //ELIS2_0_UTILS_H

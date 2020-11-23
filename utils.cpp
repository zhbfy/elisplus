//
// Created by house on 18-6-1.
//

//#include "types.h"
#include "utils.h"
//#include "stdfxa.h"
#include <iostream>
#include <fstream>
#include <string>
#include<direct.h>
#include <windows.h>

using namespace std;

void SplitStringToDoubleArray(const string &s, const string &c, vector<double> &v) {
    string::size_type pos1 = 0, pos2 = s.find(c);
    while (string::npos != pos2) {
        v.push_back(stod(s.substr(pos1, pos2 - pos1)));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length()) {
        v.push_back(stod(s.substr(pos1)));
    }
}

void ReadData(const string &file_name, mydata &f) {
    f.data.clear();
    f.label.clear();

    set<int> uniq;
    uniq.clear();

	ifstream file;
	file.open(file_name, ios::binary);
    string s;
    string c = "\t";
    while (getline(file,s)) {
        // read label
        string::size_type pos = s.find(c);
        int label = stoi(s.substr(0, pos));
        f.label.push_back(label);
        uniq.insert(label);
        s = s.substr(pos + c.size());

        // read data
        vector<double> tmp;
        tmp.clear();
        SplitStringToDoubleArray(s, c, tmp);
        f.data.push_back(tmp);
    }

    f.num = f.data.size();
    f.len = f.data[0].size();
    f.classes = uniq.size();

    auto order = new int[f.classes];
    int i = 0;
    for (auto it = uniq.begin(); it != uniq.end(); ++i, ++it) {
        order[i] = *it;
    }
    for (i = 0; i < f.num; ++i) {
        f.label[i] = (int) (lower_bound(order, order + f.classes, f.label[i]) - order);
    }
    delete[] order;

    file.close();
}

void App_data(mydata &f, mydata &q, int window) {
	q.label.clear();
	q.data.clear();
	q.label = f.label;
	
	for (int i = 0; i < f.num; ++i) {
		vector <double> v;
		
		for (int j = 0; j < ceil(f.len / window); ++j) {
			double tmp=0;
			if ((j+1)*window <= f.num) {
				for (int k = 0; k < window; ++k) {
					tmp += f.data[i][j*window + k];
				}
				tmp /= window;
			}
			else {
				for (int k = 0; k < f.len-j*window; ++k) {
					tmp += f.data[i][j*window + k];
				}
				tmp /= (f.len - j * window);
			}
			v.push_back(tmp);
		}
		q.data.push_back(v);
	}
	q.num = f.num;
	q.len = q.data[0].size();
	q.classes = f.classes;
}

void App_data_smooth(mydata &f, mydata &q, int window) {
	q.label.clear();
	q.data.clear();
	q.label = f.label;

	for (int i = 0; i < f.num; ++i) {
		vector <double> v;

		for (int j = 0; j < ceil(f.len / (window/2)); ++j) {
			double tmp = 0;
			if (j *(window/2)+ window <= f.num) {
				for (int k = 0; k < window; ++k) {
					tmp += f.data[i][j*(window/2) + k];
				}
				tmp /= window;
			}
			else {
				for (int k = 0; k < f.len - j * (window/2); ++k) {
					tmp += f.data[i][j*(window/2) + k];
				}
				tmp /= (f.len - j * (window/2));
			}
			v.push_back(tmp);
		}
		q.data.push_back(v);
	}
	q.num = f.num;
	q.len = q.data[0].size();
	q.classes = f.classes;
}

void ReadELIS(const bool training, ELIS &elis) {
    for (int i = 0; i < elis.classes; ++i) {
        elis.shapelets[i].shape.clear();
        elis.shapelets[i].len.clear();
        elis.regressions[i].w.clear();
    }


	//ifstream file(training ? "D:\\code\\supervisedshapeletbasedtimeseriesclassification\\ELIS-master\\ELIS-master\\eigen_datagen\\CBF\\init.txt" : "D:\\code\\supervisedshapeletbasedtimeseriesclassification\\ELIS-master\\ELIS-master\\eigen_datagen\\CBF\\learned.txt");
    ifstream file(training ? "init.txt" : "learned.txt");
    string s;
    string c = ",";

    // w0
    if (training) {
        for (int i = 0; i < elis.classes; ++i) {
            elis.regressions[i].w0 = 1;
        }
    } else {
        file >> s;
        vector<double> tmp;
        tmp.clear();
        SplitStringToDoubleArray(s, c, tmp);
        for (int i = 0; i < elis.classes; ++i) {
            elis.regressions[i].w0 = tmp[i];
        }
    }

    for (int i = 0; i < elis.classes; ++i) {
        file >> s;
        elis.numbers[i] = stoul(s);
        for (int j = 0; j < elis.numbers[i]; ++j) {
            file >> s;

            // w
            string::size_type pos1 = s.find(c);
            elis.regressions[i].w.push_back(stod(s.substr(0, pos1)));

            // len
            string::size_type pos2 = s.find(c, pos1 + c.size());
            elis.shapelets[i].len.push_back(stoi(s.substr(pos1 + c.size(), pos2 - pos1 - c.size())));

            // shapelet
            s = s.substr(pos2 + c.size());
            vector<double> tmp;
            SplitStringToDoubleArray(s, c, tmp);
            elis.shapelets[i].shape.push_back(tmp);
        }
    }

    file.close();
}

void App_shapelet(ELIS &elis, ELIS &elis1, int window) {
	for (int i = 0; i < elis1.classes; ++i) {
		elis1.shapelets[i].shape.clear();
		elis1.shapelets[i].len.clear();
		elis1.regressions[i].w.clear();
	}
	elis1.classes = elis.classes;
	elis1.numbers = new size_t[elis.classes];
	elis1.shapelets = new shapelet[elis.classes];
	elis1.regressions = new regression[elis.classes];
	
	for (int i = 0; i < elis.classes; ++i) {
		elis1.regressions[i].w0 = elis.regressions[i].w0;
		elis1.numbers[i] = elis.numbers[i];
		for (int j = 0; j < elis1.numbers[i]; ++j) {
			vector<double> v;
			elis1.regressions[i].w.push_back(elis.regressions[i].w[j]);
			elis1.shapelets[i].len.push_back(ceil(elis.shapelets[i].len[j]/window));
			for (int k = 0; k < elis1.shapelets[i].len[j]; k++) {
				double tmp = 0;
				if ((k+1)*window <= elis.shapelets[i].len[j]) {
					for (int h = 0; h < window; h++) {
						//printf("%d,%d,%d,%d,%d,%d\n", i, j, k, h, elis.shapelets[i].len[j], elis1.shapelets[i].len[j]);
						tmp += elis.shapelets[i].shape[j][k*window + h];
					}
					tmp /= window;
				}
				else {
					for (int h = 0; h < elis.shapelets[i].len[j]-k*window; h++) {
						tmp += elis.shapelets[i].shape[j][k*window + h];
					}
					tmp /= elis.shapelets[i].len[j] - k * window;
				}
				v.push_back(tmp);
				//elis1.shapelets[i].shape[j].push_back(tmp);
			}
			elis1.shapelets[i].shape.push_back(v);
		}
	}
}

void App_shapelet_smooth(ELIS &elis, ELIS &elis1, int window) {
	for (int i = 0; i < elis1.classes; ++i) {
		elis1.shapelets[i].shape.clear();
		elis1.shapelets[i].len.clear();
		elis1.regressions[i].w.clear();
	}
	elis1.classes = elis.classes;
	elis1.numbers = new size_t[elis.classes];
	elis1.shapelets = new shapelet[elis.classes];
	elis1.regressions = new regression[elis.classes];

	for (int i = 0; i < elis.classes; ++i) {
		elis1.regressions[i].w0 = elis.regressions[i].w0;
		elis1.numbers[i] = elis.numbers[i];
		for (int j = 0; j < elis1.numbers[i]; ++j) {
			vector<double> v;
			elis1.regressions[i].w.push_back(elis.regressions[i].w[j]);
			elis1.shapelets[i].len.push_back(ceil(elis.shapelets[i].len[j] / (window/2)));
			for (int k = 0; k < elis1.shapelets[i].len[j]; k++) {
				double tmp = 0;
				if ((k)*(window/2)+window <= elis.shapelets[i].len[j]) {
					for (int h = 0; h < window; h++) {
						//printf("%d,%d,%d,%d,%d,%d\n", i, j, k, h, elis.shapelets[i].len[j], elis1.shapelets[i].len[j]);
						tmp += elis.shapelets[i].shape[j][k*(window/2) + h];
					}
					tmp /= window;
				}
				else {
					for (int h = 0; h < elis.shapelets[i].len[j] - k * (window/2); h++) {
						tmp += elis.shapelets[i].shape[j][k*(window/2) + h];
					}
					tmp /= elis.shapelets[i].len[j] - k * (window/2);
				}
				v.push_back(tmp);
				//elis1.shapelets[i].shape[j].push_back(tmp);
			}
			elis1.shapelets[i].shape.push_back(v);
		}
	}
}

void WriteModel(const ELIS &elis, const bool discovering) {
    ofstream file(discovering ? "init.txt" : "learned.txt");
    string c = ",";

    if (!discovering) {
        // w0
        file << elis.regressions[0].w0;
        for (int i = 1; i < elis.classes; ++i) {
            file << "," << elis.regressions[i].w0;
        }
        file << endl;
    }

    for (int i = 0; i < elis.classes; ++i) {
        file << elis.numbers[i] << endl;
        for (int j = 0; j < elis.numbers[i]; ++j) {
            // w and len
            file << elis.regressions[i].w[j] << "," << elis.shapelets[i].len[j];

            // shapelet
            for (int k = 0; k < elis.shapelets[i].len[j]; ++k) {
                file << "," << elis.shapelets[i].shape[j][k];
            }
            file << endl;
        }
    }

    file.close();
}

void WriteModel_dis(const ELIS &elis, const bool discovering) {
	
	

	//ofstream file(discovering ? "init.txt" + to_string(suffix) : "learned.txt");
	ofstream file(discovering ? "init.txt" : "learned.txt");
	string c = ",";

	if (!discovering) {
		// w0
		file << elis.regressions[0].w0;
		for (int i = 1; i < elis.classes; ++i) {
			file << "," << elis.regressions[i].w0;
		}
		file << endl;
	}

	for (int i = 0; i < elis.classes; ++i) {
		file << elis.numbers[i] << endl;
		for (int j = 0; j < elis.numbers[i]; ++j) {
			// w and len
			file << elis.regressions[i].w[j] << "," << elis.shapelets[i].len[j];

			// shapelet
			for (int k = 0; k < elis.shapelets[i].len[j]; ++k) {
				file << "," << elis.shapelets[i].shape[j][k];
			}
			file << endl;
		}
	}

	file.close();

	int suffix = 0;
	fstream _file;
	file.open("init.txt"+ to_string(suffix), ios::in);
	//while (_file) {
	for(;;){
		if (_file) {
			_file.close();
			suffix++;
		}
		else {
			break;
		}
		_file.open("init.txt" + to_string(suffix), ios::in);
	}
	string dst = "init.txt" + to_string(suffix);
	LPCSTR str = dst.c_str();
	CopyFile("init.txt",str , TRUE);
}


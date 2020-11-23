//
// Created by house on 18-6-1.
//


#include <ctime>
#include <cstring>

#include <random>
//#include "stdfxa.h"
//#include "types.h"
#include "elis.h"
#include "utils.h"


using namespace std;

mydata x, y,xp,yp;
ELIS elis,elis1;

void GenData(int rep) {
	size_t tmpn = x.num;
	random_device rd{};
	mt19937 gen1{ rd() };
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

	string operation = argv[1];
	string name = argv[2];
	int window = stoi(argv[3]);

	//int gen1 = stoi(argv[4]);
	
	//string operation = "train";
	//string name = "D:\\code\\supervisedshapeletbasedtimeseriesclassification\\UCRArchive_2018\\UCRArchive_2018\\CBF\\CBF_TRAIN.tsv";
	//int window = 4;
	//int gen1 = stoi(argv[4]);

	ReadData(name, x);
	printf("0");
	GenData(0);
	printf("1");
	App_data(x, xp, window);
	printf("2");
	elis.classes = x.classes;
	elis.numbers = new size_t[elis.classes];
	elis.shapelets = new shapelet[elis.classes];
	elis.regressions = new regression[elis.classes];

	printf("3");
	if (operation == "train") {
		ReadELIS(true, elis);
		App_shapelet(elis, elis1, window);
		if (argc == 8) {
		//if (1) {
			ReadData(string(argv[7]), y);
			//ReadData(string("D:\\code\\supervisedshapeletbasedtimeseriesclassification\\UCRArchive_2018\\UCRArchive_2018\\CBF\\CBF_TEST.tsv"), y);
			App_data(y, yp, window);
			parameters fzc(stod(string(argv[4])), stoi(string(argv[5])), stod(string(argv[6])));
			//parameters fzc(stod(string("0.01")), stoi(string("1000")), stod(string("0.01")));
			Train(xp, xp, fzc, elis1);
		}
		else {
			parameters fzc(stod(string(argv[3])), stoi(string(argv[4])), stod(string(argv[5])));
			Train(x, fzc, elis);
		}
		WriteModel(elis1, false);
	}
	else {
		ReadELIS(false, elis);
		Test(x, elis);
	}

	delete[] elis.numbers;
	delete[] elis.shapelets;
	delete[] elis.regressions;

	double end_time = clock();
	printf("Total Running Time = %.3f sec\n", (end_time - start_time) / CLOCKS_PER_SEC);

	return 0;
}
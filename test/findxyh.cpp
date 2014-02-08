#include <xyh.hpp>
#include <iostream>
#include <cmath>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

using namespace std;

const double ERROR_ALLOWANCE = 2.0;
const int TARGET_COUNT = 8;
const int CAMERA_COUNT = 3;
struct FindXYHTest {
	int R[TARGET_COUNT];
	int P[CAMERA_COUNT][TARGET_COUNT];
	double xR;
	double yR;
	double heading;
};

void fillP(int array[][TARGET_COUNT], int numRows, int numCols) {
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			if (array[i][j] == 0) {//*(array + i * numCols + j) == 0) { //if (array[i][j] == 0) { 
				//*(array + i*numCols + j) = -1;//
				array[i][j] = -1;
			}
		}
	}
}

FindXYHTest tests[] = {
	{{283, 276, 449, 453}, {{0, 320}, {0}, {0}}, 224.01, 153.89, 356.06},
	{{283, 276, 449, 453}, {{178}, {0}, {0}}, 224.01, 153.89, 315.28},
	{{283, 276, 449, 453}, {{632}, {0}, {0}}, 224.01, 153.89, 272.72},
	{{283, 276, 449, 453}, {{1026}, {0}, {0}}, 224.01, 153.89, 235.78},
	{{283, 276, 449, 453}, {{0, 0, 0, 226}, {0}, {0}}, 224.01, 153.89, 175.89},
	{{283, 276, 449, 453}, {{0, 0, 329}, {0}, {0}}, 224.01, 153.89, 133.49},
	{{0, 0, 449, 453}, {{0, 0, 329}, {0}, {0}}, 222.08, 155.10, 133.65},
	{{270, 181, 504, 543}, {{0}, {0, 0, 569}}, 159.99, 233.66, 1.94},
	{{270, 181, 504, 543}, {{0}, {0, 0, 959}}, 159.99, 233.66, 325.38},
	{{270, 181, 504, 543}, {{0}, {0, 457}}, 159.99, 233.66, 210.40},
	{{270, 181, 504, 543}, {{0}, {60, 0}}, 159.99, 233.66, 182.37},
	{{270, 181, 504, 543}, {{0}, {563, 0}}, 159.99, 233.66, 135.21},
	{{270, 0, 0, 543}, {{0}, {563, 0}}, 156.23, 236.15, 134.35},
	{{137, 0, 0, 0, 136}, {{296, 0, 0, 0, 342}, {0}, {0}}, 0, 0, 0}
};

int main() {
	int ret = 0;
	for (FindXYHTest &test : tests) {
		fillP(test.P, CAMERA_COUNT, TARGET_COUNT);
		double xR, yR, heading;
		FindXYH(test.R[0], test.R[1], test.R[2], test.R[3], test.R[4],
			test.R[5], test.R[6], test.R[7], test.P, xR, yR, heading);
		if (abs(test.xR - xR) > ERROR_ALLOWANCE
			|| abs(test.yR - yR) > ERROR_ALLOWANCE
			|| abs(test.heading - heading) > ERROR_ALLOWANCE) {
			cout << ANSI_COLOR_RED << "[FAIL] ";
			ret = 1;
		} else {
			cout << ANSI_COLOR_GREEN << "[PASS] ";
		}
		cout << "FindXYH(" << test.R[0] << ", " << test.R[1] << ", ";
		cout << test.R[2] << ", " << test.R[3] << ", " << test.R[4]  << ", ";
		cout << test.R[5] << ", " << test.R[6] << ", " << test.R[7] << ")";
		cout << ": " << xR << "xR, " << yR << "yR, " << heading << "H, ";
		cout << "expected: " << test.xR << "xR, " << test.yR << "yR, ";
		cout << heading << "H." << ANSI_COLOR_RESET << endl;
	}
	return ret;
}


#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <vector>

using namespace std;

int main() {

    string line;
    ifstream inFile;
    inFile.open("/home/caelinux2/Desktop/trunk/scripts/ReadableOutput.txt");
    std::vector<double[]> data;
    int idx = 0;
    double arr[1000];

    if (inFile.is_open()) {
        while (getline(inFile, line)) {
            string::size_type loc = line.find("data", 0);
            if (loc != string::npos) {

                string temp = line.substr(line.find("=") + 2);
                arr[idx++] = atof(temp.c_str());
            }
            if (line.find("===================================================") != string::npos && idx != 0) {
                data.push_back(arr);  /*To that list, adding the array*/
                idx = 0; /*re-initialize the array*/
                memset(arr, 0, sizeof(arr)); /*clearing up the array, ready for next iteration*/
            }
        }

    }

            else cout << "Unable to open file";

        inFile.close();
        return 0;

    }

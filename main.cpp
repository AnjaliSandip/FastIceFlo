#include <iostream>
#include <fstream>
using namespace std;

int main() {

    string line;
    ifstream inFile;
    inFile.open("/home/caelinux2/Desktop/trunk/scripts/ReadableOutput.txt");

    if (inFile.is_open()) {
        while (getline(inFile, line)) {
            cout << line << '\n';
        }

        inFile.close();

    }

    else cout << "Unable to open file";

    /*Finite element/mesh properties*/
    int index [614][3] ={1};
    int x [340] = {1};
    int y [340] ={1};
    const int nbv =340;  /*total number of unique nodes*/
    const int nbe = 614;


    /*Constants*/
    int n_glen = 3;
    int damp = 2;
    double rele = 1e-1;
    double eta_b = 0.5;
    double eta_0 = 1.e+14 / 2.;
    double niter = 5e6;
    int nout_iter = 1000;
    double epsi = 1e-8;

    /*Initial guesses */
    int vx [nbv] = {0};
    int vy [nbv] = {0};
    double etan [nbe] = {1e14};
    int dvxdt [nbv] = {0};
    int dvydt [nbv] = {0};

    /*Calculate Nodal coefficients and areas*/
    double one = 1;
    int k = 0;
    int n1;
    int n2;
    int n3;
    int t_nbv = nbe *3;  /*total number of nodes = number of elements * 3*/
    double x1;
    double x2;
    double x3;
    double y1;
    double y2;
    double y3;
    double invdet;
    double alpha [nbe][3];
    double beta [nbe][3];
    double areas [nbe];
    double areasAll[t_nbv][2];

    for (int i = 0; i < 615; i++) {
        n1 = index[i][0];
        n2 = index[i][1];
        n3 = index[i][2];
        x1 = x[n1-1];
        x2 = x[n2-1];
        x3 = x[n3-1];
        y1 = y[n1-1];
        y2 = y[n2-1];
        y3 = y[n3-1];

        invdet = one/(x1*(y2-y3)-x2*(y1-y3)+x3*(y1-y2));

        alpha[i][0] = invdet*(y2-y3);
        alpha[i][1] = invdet*(y2-y3);
        alpha[i][2] =invdet*(y1-y2);

        beta[i][0] = invdet*(x3-x2);
        beta[i][1] = invdet*(x1-x3);
        beta[i][2] = invdet*(x2-x1);

        areas[i] = (0.5*((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1)));

        areasAll[k][0] = {(0.5*((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1)))};
        areasAll[k+1][0] = {(0.5*((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1)))};
        areasAll[k+2][0] = {(0.5*((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1)))};

        areasAll[k][1] = n1;
        areasAll[k+1][1] = n2;
        areasAll[k+2][1] = n3;

        k = k+3;

    }

    /* Calculate Nodal weights*/
    double Weights ;
    double NodalWeights [t_nbv][2];
    int nodes [t_nbv];
    double n;

    for (int i = 0; i < t_nbv; i++) {
        double n = areasAll[i][1];


        for (int l = 0; l < i; l++) {
            if (n == nodes[l]) {
                goto LOOP;
            }

        }

        for (int j = 0; j < t_nbv; j++) {
            if (areasAll[j][1] == n) {
                Weights = Weights + areasAll[j][0];
                printf("%f\n", areasAll[j][0]);
            }
        }

        nodes[i] = n;
        NodalWeights [i][0] = n;
        printf("%f\n", NodalWeights[i][0]);
        NodalWeights [i][1] = Weights;
        printf("%f\n", NodalWeights[i][1]);
        Weights = 0;
        LOOP:;
    }
    return 0;

}
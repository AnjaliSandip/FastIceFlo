#include <iostream>
#include <fstream>
using namespace std;


/*Finite element/mesh properties*/
const int nbv =340;  /*total number of unique nodes*/
const int nbe = 614;
const int t_nbv = nbe *3;  /*total number of nodes = number of elements * 3*/
int index [nbe][3] ={1};
int x [nbv] = {1};
int y [nbv] ={1};

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

/*Manage Derivatives once for all*/
double one = 1;
int k = 0;
int n1;
int n2;
int n3;
double x1;
double x2;
double x3;
double y1;
double y2;
double y3;
double invdet;
double alpha [nbe][3];
double beta [nbe][3];
double areas[t_nbv][2];

double NodalCoeffs(double field1[nbv], double field2[nbv]) {
    for (int i = 0; i < nbe; i++) {
        n1 = index[i][0];
        n2 = index[i][1];
        n3 = index[i][2];


        x1 = field1[n1 - 1];
        x2 = field1[n2 - 1];
        x3 = field1[n3 - 1];


        y1 = field2[n1 - 1];
        y2 = field2[n2 - 1];
        y3 = field2[n3 - 1];

        invdet = one / (x1 * (y2 - y3) - x2 * (y1 - y3) + x3 * (y1 - y2));

        alpha[i][0] = invdet * (y2 - y3);
        alpha[i][1] = invdet * (y2 - y3);
        alpha[i][2] = invdet * (y1 - y2);

        beta[i][0] = invdet * (x3 - x2);
        printf("%f\n", beta[i][0]);
        beta[i][1] = invdet * (x1 - x3);
        printf("%f\n", beta[i][1]);
        beta[i][2] = invdet * (x2 - x1);
        printf("%f\n", beta[i][2]);


        areas[k][0] = {(0.5 * ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)))};
        areas[k + 1][0] = {(0.5 * ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)))};
        areas[k + 2][0] = {(0.5 * ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)))};

        areas[k][1] = n1;
        areas[k + 1][1] = n2;
        areas[k + 2][1] = n3;

        k = k + 3;


    }
}


double SumOfAreas ;
double NodalWeights [t_nbv][2];
int nodes [t_nbv];
double n;

double Weights() {
    for (int i = 0; i < t_nbv; i++) {
        double n = areas[i][1];
/*printf("%f\n", n);*/

        for (int l = 0; l < i; l++) {
            if (n == nodes[l]) {
                goto LOOP;
            }

        }

        for (int j = 0; j < t_nbv; j++) {
            if (areas[j][1] == n) {
                SumOfAreas = SumOfAreas + areas[j][0];
                printf("%f\n", areas[j][0]);
            }
        }

        nodes[i] = n;
        NodalWeights[i][0] = n;
        printf("%f\n", NodalWeights[i][0]);
        NodalWeights[i][1] = SumOfAreas;
        printf("%f\n", NodalWeights[i][1]);
        SumOfAreas = 0;
        LOOP:;
    }
}




double derive_x_elem (double field[nbv]) {
    double field_nodes[nbv][2];    /*field_nodes: Matrix that stores the node numbers and their corresponding field values, matrix size = [nbv][2]*/
    double field_elem [nbe][3];    /*field_elem: Index matrix, except that now in place of node numbers, the corresponding field values are stored, matrix size = [nbe][3]*/
    for (int i = 0; i < nbv; i++) {
        field_nodes[i][0] = i;
        field_nodes[i][1] = field[i];
    }

    for (int i = 0; i < nbe; i++) {
        for (int j = 0; j < 3; j++){
            int node = index[i][j];
            for (int k = 0; k < nbv; k++)
            {
                if (field_nodes[k][0] == node)      /*once data has been parsed, array1, should be replaced with field_nodes*/
                {
                    field_elem [i][j]= field_nodes[k][1];
                    /*printf("%f\n",field_elem[i][j]);*/
                    break;
                }
            }
        }

    }
    double derive_fieldx_elem[nbe];   /*derive_fieldx_elem: Matrix that stores the field derivatives wrt x (on an element basis), matrix size = [nbe][1]*/
    for (int i = 0; i < nbe; i++) {
        derive_fieldx_elem[i] = field_elem[i][0]*alpha[i][0] + field_elem[i][1]*alpha[i][1] + field_elem[i][2]*alpha[i][2];

    }

}

double derive_y_elem (double field[nbv]) {
    double field_nodes[nbv][2];    /*field_nodes: Matrix that stores the node numbers and their corresponding field values, matrix size = [nbv][2]*/
    double field_elem [nbe][3];    /*field_elem: Index matrix, except that now in place of node numbers, the corresponding field values are stored, matrix size = [nbe][3]*/
    for (int i = 0; i < nbv; i++) {
        field_nodes[i][0] = i;
        field_nodes[i][1] = field[i];
    }


    for (int i = 0; i < nbe; i++) {
        for (int j = 0; j < 3; j++){
            int node = index[i][j];
            for (int k = 0; k < nbv; k++)
            {
                if (field_nodes[k][0] == node)
                {
                    field_elem [i][j]= field_nodes[k][1];
                    printf("%f\n",field_elem[i][j]);
                    break;
                }
            }
        }

    }
    double derive_fieldy_elem[nbe];   /*derive_fieldy_elem: Matrix that stores the field derivatives wrt y (on an element basis), matrix size = [nbe][1]*/
    for (int i = 0; i < nbe; i++) {

        derive_fieldy_elem[i] = field_elem[i][0]*beta[i][0] + field_elem[i][1]*beta[i][1] + field_elem[i][2]*beta[i][2];

    }

}


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

    NodalCoeffs(x, y);

    Weights();
    derive_x_elem(vx);
    derive_y_elem(vy);



    return 0;

}

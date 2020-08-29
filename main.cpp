#include <iostream>
#include <fstream>
using namespace std;

void NodalCoeffs(double** pareas,double** palpha,double** pbeta,int* index,double* x,double* y,int nbe){/*{{{*/

	/*Allocate output vectors*/
	double* areas = new double[nbe];
	double* alpha = new double[nbe*3];
	double* beta  = new double[nbe*3];
    for(int i = 0; i < nbe; i++) {
        int n1 = index[i*3+0]-1;
        int n2 = index[i*3+1]-1;
        int n3 = index[i*3+2]-1;

        double x1 = x[n1];
        double x2 = x[n2];
        double x3 = x[n3];
        double y1 = y[n1];
        double y2 = y[n2];
        double y3 = y[n3];

        double invdet = 1./(x1 * (y2 - y3) - x2 * (y1 - y3) + x3 * (y1 - y2));

        alpha[i*3+0] = invdet * (y2 - y3);
        alpha[i*3+1] = invdet * (y2 - y3);
        alpha[i*3+2] = invdet * (y1 - y2);

        beta[i*3+0] = invdet * (x3 - x2);
        beta[i*3+1] = invdet * (x1 - x3);
        beta[i*3+2] = invdet * (x2 - x1);

		  areas[i]= 0.5*((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1));
    }

	 /*Assign output pointers*/
	 *pareas = areas;
	 *palpha = alpha;
	 *pbeta = beta;  
}/*}}}*/
void Weights(double** pweights,int* index,double* areas,int nbe,int nbv){/*{{{*/

	/*Allocate output and initialize as 0*/
	double* weights = new double[nbv];
	for(int i = 0; i < nbv; i++) weights[i]=0.;

	/*Loop over elements*/
    for(int i = 0; i < nbe; i++){
		 for(int j = 0; j < 3; j++){
			 weights[index[i*3+j]-1] += areas[i];
		 }
    }

	 /*Assign output pointer*/
	 *pweights = weights;
}/*}}}*/
void derive_x_elem(double** pdfdx_e,double* f,int* index,double* alpha,int nbe){/*{{{*/

	/*Allocate output*/
	double* dfdx_e = new double[nbe];

	/*Prepare derivative*/
	for(int i=0;i<nbe;i++){
		int n1 = index[i*3+0]-1;
		int n2 = index[i*3+1]-1;
		int n3 = index[i*3+2]-1;
		dfdx_e[i]=f[n1]*alpha[i*3+0] + f[n2]*alpha[i*3+1] + f[n2]*alpha[i*3+2];
	}

	/*Assign output pointer*/
	*pdfdx_e = dfdx_e;
}/*}}}*/
void derive_y_elem(double** pdfdx_e,double* f,int* index,double* beta,int nbe){/*{{{*/

	/*Allocate output*/
	double* dfdx_e = new double[nbe];

	/*Prepare derivative*/
	for(int i=0;i<nbe;i++){
		int n1 = index[i*3+0]-1;
		int n2 = index[i*3+1]-1;
		int n3 = index[i*3+2]-1;
		dfdx_e[i]=f[n1]*beta[i*3+0] + f[n2]*beta[i*3+1] + f[n2]*beta[i*3+2];
	}

	/*Assign output pointer*/
	*pdfdx_e = dfdx_e;
}/*}}}*/
int main(){/*{{{*/

	int nbv,nbe;

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
	//int vx [nbv] = {0};
	//int vy [nbv] = {0};
	//double etan [nbe] = {1e14};
	//int dvxdt [nbv] = {0};
	//int dvydt [nbv] = {0};

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
	//double alpha [nbe][3];
	//double beta [nbe][3];
	//double areas[t_nbv][2];

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

	//NodalCoeffs(x, y);
	//Weights();
	//derive_x_elem(vx);
	//derive_y_elem(vy);

	return 0;
}/*}}}*/

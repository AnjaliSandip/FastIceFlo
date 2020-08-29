#include <iostream>
#include <fstream>
using namespace std;

/*I/O stuff {{{*/
FILE* SetFilePointerToData(FILE* fid,int* pcode,int* pvector_type,const char* data_name){/*{{{*/

	int found  = 0;
	const char* mddot = "md.";
	char* record_name = NULL;
	int   record_name_size;
	long long record_length;
	int record_code;       //1 to 7 number
	int vector_type   = 0; //nodal or elementary

	if(strncmp(data_name,mddot,3)!=0){
		std::cerr <<"Cannot fetch \""<<data_name<<"\" does not start with \""<<mddot<<"\"";
	}

		/*First set FILE* position to the beginning of the file: */
		fseek(fid,0,SEEK_SET);

		/*Now march through file looking for the correct data identifier: */
		for(;;){
			/*Read size of first string name: */
			if(fread(&record_name_size,sizeof(int),1,fid)==0){
				/*we have reached the end of the file. break: */
				delete record_name;
				break;
			}
			if(record_name_size<3 || record_name_size>80){
				std::cerr<<"error while looking in binary file. Found a string of size "<<record_name_size;
			}

			/*Allocate string of correct size: */
			record_name=new char[record_name_size+1];
			record_name[record_name_size]='\0';

			/*Read record_name: */
			if(fread(record_name,record_name_size*sizeof(char),1,fid)==0){
				/*we have reached the end of the file. break: */
				found=0;
				delete record_name;
				break;
			}
			if(strncmp(record_name,mddot,3)!=0){
				std::cerr<<"error while reading binary file: record does not start with \"md.\": "<<record_name;
			}

			/*Is this the record sought for? : */
			if(strcmp(record_name,data_name)==0){
				/*Ok, we have found the correct string. Pass the record length, and read data type code: */
				fseek(fid,sizeof(long long),SEEK_CUR);
				if(fread(&record_code,sizeof(int),1,fid)!=1) std::cerr<<"Could not read record_code";
				/*if record_code points to a vector, get its type (nodal or elementary): */
				if((5<=record_code && record_code<=7) || record_code==10){
					if(fread(&vector_type,sizeof(int),1,fid)!=1) std::cerr<<"Could not read vector_type";
				}
				found=1;
				delete record_name;
				break;
			}
			else{
				/*This is not the correct string, read the record length, and use it to skip this record: */
				if(fread(&record_length,sizeof(long long),1,fid)!=1) std::cerr<<"Could not read record_length";
				/*skip: */
				fseek(fid,record_length,SEEK_CUR);
				delete record_name;
			}
		}
	if(!found) std::cerr<<"could not find data with name \"" << data_name << "\" in binary file";

	/*Assign output pointers:*/
	*pcode=record_code;
	if(pvector_type) *pvector_type=vector_type;

	return fid;
}
/*}}}*/
void  FetchData(FILE* fid,int* pinteger,const char* data_name){/*{{{*/

	/*output: */
	int integer;
	int code;

	/*Set file pointer to beginning of the data: */
	fid=SetFilePointerToData(fid,&code,NULL,data_name);

	if(code!=2)std::cerr <<"expecting an integer for \"" << data_name<<"\"";

	/*We have to read a integer from disk. First read the dimensions of the integer, then the integer: */
	if(fread(&integer,sizeof(int),1,fid)!=1) std::cerr<<"could not read integer ";

	/*Assign output pointers: */
	*pinteger=integer;
}/*}}}*/
void  FetchData(FILE* fid,int** pmatrix,int* pM,int* pN,const char* data_name){/*{{{*/

	/*output: */
	int M,N;
	double* matrix=NULL;
	int* integer_matrix=NULL;
	int code=0;

	/*Set file pointer to beginning of the data: */
	fid=SetFilePointerToData(fid,&code,NULL,data_name);
	if(code!=5 && code!=6 && code!=7)std::cerr<<"expecting a IssmDouble, integer or boolean matrix for \""<<data_name<<"\""<<" (Code is "<<code<<")";

	/*Now fetch: */

	/*We have to read a matrix from disk. First read the dimensions of the matrix, then the whole matrix: */
	/*numberofelements: */
	if(fread(&M,sizeof(int),1,fid)!=1) std::cerr<<"could not read number of rows for matrix ";
	if(fread(&N,sizeof(int),1,fid)!=1) std::cerr<<"could not read number of columns for matrix ";

	/*Now allocate matrix: */
	if(M*N){
		matrix=new double[M*N];

		/*Read matrix on node 0, then broadcast: */
		if(fread(matrix,M*N*sizeof(double),1,fid)!=1) std::cerr<<"could not read matrix ";
	}

	/*Now cast to integer: */
	if(M*N){
		integer_matrix=new int[M*N];
		for (int i=0;i<M;i++){
			for (int j=0;j<N;j++){
				integer_matrix[i*N+j]=(int)matrix[i*N+j];
			}
		}
	}
	else{
		integer_matrix=NULL;
	}
	/*Free ressources:*/
	delete matrix;

	/*Assign output pointers: */
	*pmatrix=integer_matrix;
	if(pM)*pM=M;
	if(pN)*pN=N;
}/*}}}*/
void  FetchData(FILE* fid,double** pmatrix,int* pM,int* pN,const char* data_name){/*{{{*/

	/*output: */
	int M,N;
	double* matrix=NULL;
	int* integer_matrix=NULL;
	int code=0;

	/*Set file pointer to beginning of the data: */
	fid=SetFilePointerToData(fid,&code,NULL,data_name);
	if(code!=5 && code!=6 && code!=7)std::cerr<<"expecting a IssmDouble, integer or boolean matrix for \""<<data_name<<"\""<<" (Code is "<<code<<")";

	/*Now fetch: */

	/*We have to read a matrix from disk. First read the dimensions of the matrix, then the whole matrix: */
	/*numberofelements: */
	if(fread(&M,sizeof(int),1,fid)!=1) std::cerr<<"could not read number of rows for matrix ";
	if(fread(&N,sizeof(int),1,fid)!=1) std::cerr<<"could not read number of columns for matrix ";

	/*Now allocate matrix: */
	if(M*N){
		matrix=new double[M*N];

		/*Read matrix on node 0, then broadcast: */
		if(fread(matrix,M*N*sizeof(double),1,fid)!=1) std::cerr<<"could not read matrix ";
	}

	/*Assign output pointers: */
	*pmatrix=matrix;
	if(pM)*pM=M;
	if(pN)*pN=N;
}/*}}}*/
/*}}}*/
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

	/*Open input binary file*/
	const char* inputfile = "/Users/mmorligh/Desktop/issmuci/trunk-jpl/execution/test101-08-24-2020-08-42-58-1175/test101.bin";
	FILE* fid = fopen(inputfile,"rb");
   if(fid==NULL) std::cerr<<"could not open file " << inputfile << " for binary reading or writing";

	/*Get Mesh properties*/
	int nbe,nbv,M,N;
	int *index;
	double* x=NULL;
	double* y=NULL;
	FetchData(fid,&nbe,"md.mesh.numberofelements");
	FetchData(fid,&nbv,"md.mesh.numberofvertices");
	FetchData(fid,&index,&M,&N,"md.mesh.elements");
	FetchData(fid,&x,&M,&N,"md.mesh.x");
	FetchData(fid,&y,&M,&N,"md.mesh.y");

	/*Get initial guess*/
	double* vx = NULL;
	double* vy = NULL;
	FetchData(fid,&vx,&M,&N,"md.initialization.vx");
	FetchData(fid,&vy,&M,&N,"md.initialization.vy");

	/*ETC, do the same with rheology_B etc*/

	/*Close input file*/
	if(fclose(fid)!=0) std::cerr<<"could not close file " << inputfile;

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


	//NodalCoeffs(x, y);
	//Weights();
	//derive_x_elem(vx);
	//derive_y_elem(vy);

	return 0;
}/*}}}*/

#include <iostream>
#include <string.h>
#include <cmath>
using namespace std;

/*define GPU specific vairables*/
#define blockSize 1024
#define gridSize  32

/*CPU Code*/
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
			delete [] record_name;
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
			delete [] record_name;
			break;
		}
		else{
			/*This is not the correct string, read the record length, and use it to skip this record: */
			if(fread(&record_length,sizeof(long long),1,fid)!=1) std::cerr<<"Could not read record_length";
			/*skip: */
			fseek(fid,record_length,SEEK_CUR);
			delete [] record_name;
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
	delete [] matrix;

	/*Assign output pointers: */
	*pmatrix=integer_matrix;
	if(pM)*pM=M;
	if(pN)*pN=N;
}/*}}}*/
void  FetchData(FILE* fid,double* pdouble,const char* data_name){/*{{{*/

	/*output: */
	double value;
	int code;

	/*Set file pointer to beginning of the data: */
	fid=SetFilePointerToData(fid,&code,NULL,data_name);

	if(code!=3)std::cerr <<"expecting a double for \"" << data_name<<"\"";

	/*We have to read a integer from disk. First read the dimensions of the integer, then the integer: */
	if(fread(&value,sizeof(double),1,fid)!=1) std::cerr<<"could not read scalar";

	/*Assign output pointers: */
	*pdouble=value;
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
void  WriteData(FILE* fid,double* matrix,int M,int N,const char* data_name){/*{{{*/

	/*First write enum: */
	int length=(strlen(data_name)+1)*sizeof(char);
	fwrite(&length,sizeof(int),1,fid);
	fwrite(data_name,length,1,fid);

	/*Now write time and step: */
	double time = 0.;
	int    step = 1;
	fwrite(&time,sizeof(double),1,fid);
	fwrite(&step,sizeof(int),1,fid);

	/*writing a IssmDouble array, type is 3:*/
	int type=3;
	fwrite(&type,sizeof(int),1,fid);
	fwrite(&M,sizeof(int),1,fid);
	fwrite(&N,sizeof(int),1,fid);
	fwrite(matrix,M*N*sizeof(double),1,fid);
}/*}}}*/
void  WriteData(FILE* fid,const char* string,const char* data_name){/*{{{*/

	/*First write enum: */
	int length=(strlen(data_name)+1)*sizeof(char);
	fwrite(&length,sizeof(int),1,fid);
	fwrite(data_name,length,1,fid);

	/*Now write time and step: */
	double time = 0.;
	int    step = 1;
	fwrite(&time,sizeof(double),1,fid);
	fwrite(&step,sizeof(int),1,fid);

	/*writing a string, type is 2: */
	int type=2;
	fwrite(&type,sizeof(int),1,fid);

	length=(strlen(string)+1)*sizeof(char);
	fwrite(&length,sizeof(int),1,fid);
	fwrite(string,length,1,fid);
}/*}}}*/
/*}}}*/
void NodalCoeffs(double** pareas,double** palpha,double** pbeta,int* index,double* x,double* y,int nbe){/*{{{*/

	/*Allocate output vectors*/
	double* areas = new double[nbe];
	double* alpha = new double[nbe*3];
	double* beta  = new double[nbe*3];

   /*Loop over all elements and calculate nodal function coefficients and element surface area*/
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
		alpha[i*3+1] = invdet * (y3 - y1);
		alpha[i*3+2] = invdet * (y1 - y2);

		beta[i*3+0] = invdet * (x3 - x2);
		beta[i*3+1] = invdet * (x1 - x3);
		beta[i*3+2] = invdet * (x2 - x1);

		areas[i] = 0.5*((x2-x1)*(y3-y1)-(y2-y1)*(x3-x1));
	}

	/*Assign output pointers*/
	*pareas = areas;
	*palpha = alpha;
	*pbeta  = beta;
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
void derive_xy_elem(double* dfdx_e,double* dfdy_e,double* f,int* index,double* alpha,double* beta,int nbe){/*{{{*/

	/*WARNING!! Assume that dfdx_e and dfdy_e have been properly allocated*/

	for(int i=0;i<nbe;i++){
		int n1 = index[i*3+0]-1;
		int n2 = index[i*3+1]-1;
		int n3 = index[i*3+2]-1;
		dfdx_e[i] = f[n1]*alpha[i*3+0] + f[n2]*alpha[i*3+1] + f[n3]*alpha[i*3+2];
		dfdy_e[i] = f[n1]*beta[ i*3+0] + f[n2]*beta[ i*3+1] + f[n3]*beta[ i*3+2];
	}
}/*}}}*/
void elem2node(double* f_v,double* f_e,int* index,double* areas,double* weights,int nbe,int nbv){/*{{{*/

	/*WARNING!! Assume that f_v has been properly allocated*/

	/*Reinitialize output*/
	for(int i=0;i<nbv;i++) f_v[i] = 0.;

	/*Add contributions from all elements connected to vertex i*/
	for(int i=0;i<nbe;i++){
		int n1 = index[i*3+0]-1;
		int n2 = index[i*3+1]-1;
		int n3 = index[i*3+2]-1;
		f_v[n1] += f_e[i]*areas[i];
		f_v[n2] += f_e[i]*areas[i];
		f_v[n3] += f_e[i]*areas[i];
	}

	/*Divide by sum of areas*/
	for(int i=0;i<nbv;i++) f_v[i] = f_v[i]/weights[i];

}/*}}}*/
void MeshSize(double* resolx,double* resoly,int* index,double* x,double* y,double* areas,double* weights,int nbe,int nbv){/*{{{*/

	/*Get element size along x and y directions*/
	double  xmin,xmax,ymin,ymax;
	double* dx_elem = new double[nbe];
	double* dy_elem = new double[nbe];
	for(int i=0;i<nbe;i++){
		int n1 = index[i*3+0]-1;
		int n2 = index[i*3+1]-1;
		int n3 = index[i*3+2]-1;
		xmin = min(min(x[n1],x[n2]),x[n3]);
		xmax = max(max(x[n1],x[n2]),x[n3]);
		ymin = min(min(y[n1],y[n2]),y[n3]);
		ymax = max(max(y[n1],y[n2]),y[n3]);
		dx_elem[i] = xmax - xmin;
		dy_elem[i] = ymax - ymin;
	}

	/*Average over each node*/
	elem2node(resolx,dx_elem,index,areas,weights,nbe,nbv);
	elem2node(resoly,dy_elem,index,areas,weights,nbe,nbv);

	/*Cleanup and return*/
	delete [] dx_elem;
	delete [] dy_elem;
}/*}}}*/

/*CUDA Code*/
__global__ void KV(double* KVx, double* KVy, double* KVx_0, double* KVx_1, double* KVx_2, double* KVy_0, double* KVy_1, double* KVy_2, double* KVxx, double* KVyy, double* vx, double* vy, double* alpha2, double* groundedratio, double* etan, double* dvxdx, double* dvydy, double* dvxdy, double* dvydx, double* Helem, double* areas, double* alpha, double* beta, int* index, int nbv, int nbe, double* eta_nbv, double* eta_nbv_gpu, double* weights){/*{{{*/

	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if(ix<nbe){
      /*Calculate velocity derivatives*/
		dvxdx[ix] = vx[index[ix*3+0]-1]*alpha[ix*3+0] + vx[index[ix*3+1]-1]*alpha[ix*3+1] + vx[index[ix*3+2]-1]*alpha[ix*3+2];
		dvxdy[ix] = vx[index[ix*3+0]-1]*beta [ix*3+0] + vx[index[ix*3+1]-1]*beta[ ix*3+1] + vx[index[ix*3+2]-1]*beta[ ix*3+2];
		dvydx[ix] = vy[index[ix*3+0]-1]*alpha[ix*3+0] + vy[index[ix*3+1]-1]*alpha[ix*3+1] + vy[index[ix*3+2]-1]*alpha[ix*3+2];
		dvydy[ix] = vy[index[ix*3+0]-1]*beta[ ix*3+0] + vy[index[ix*3+1]-1]*beta[ ix*3+1] + vy[index[ix*3+2]-1]*beta[ ix*3+2];

      /*eta_nbv*/
      eta_nbv_gpu[ix] =  etan[ix]*areas[ix];
   }

	if(ix==0){
		for(int i=0; i<nbv; i++) eta_nbv[i] = 0.;

		for(int i=0;i<nbe;i++){
			for(int j=0; j<3;j++){
				eta_nbv[index[i*3+j]-1] += eta_nbv_gpu[i];
         }
      }

		/*Divide by sum of areas*/
		for(int i=0;i<nbv;i++) eta_nbv[i] = eta_nbv[i]/weights[i];
	}

	__syncthreads();

	/*KV*/
	double eps_xx;
	double eps_yy;
	double eps_xy;
	double eta_e;

	if(ix<nbe){
		eta_e = etan[ix];
		eps_xx = dvxdx[ix];
		eps_yy = dvydy[ix];
		eps_xy = .5 * (dvxdy[ix] + dvydx[ix]);                    

		KVx_0[ix] = 2*Helem[ix]*eta_e*(2*eps_xx + eps_yy)*alpha[ix*3+0]*areas[ix] + 2*Helem[ix]*eta_e*eps_xy*beta[ix*3+0]*areas[ix];
		KVx_1[ix] = 2*Helem[ix]*eta_e*(2*eps_xx + eps_yy)*alpha[ix*3+1]*areas[ix] + 2*Helem[ix]*eta_e*eps_xy*beta[ix*3+1]*areas[ix];
		KVx_2[ix] = 2*Helem[ix]*eta_e*(2*eps_xx + eps_yy)*alpha[ix*3+2]*areas[ix] + 2*Helem[ix]*eta_e*eps_xy*beta[ix*3+2]*areas[ix];
		KVy_0[ix] = 2*Helem[ix]*eta_e*eps_xy*alpha[ix*3+0]*areas[ix] + 2*Helem[ix]*eta_e*(2*eps_yy + eps_xx)*beta[ix*3+0]*areas[ix];
		KVy_1[ix] = 2*Helem[ix]*eta_e*eps_xy*alpha[ix*3+1]*areas[ix] + 2*Helem[ix]*eta_e*(2*eps_yy + eps_xx)*beta[ix*3+1]*areas[ix];
		KVy_2[ix] = 2*Helem[ix]*eta_e*eps_xy*alpha[ix*3+2]*areas[ix] + 2*Helem[ix]*eta_e*(2*eps_yy + eps_xx)*beta[ix*3+2]*areas[ix];
   }

	if(ix==0){
		for(int i=0; i<nbv; i++) {
			KVx[i] = 0.;
			KVy[i] = 0.; 
      }

		for(int n=0;n<nbe;n++){
			KVx[index[n*3+0]-1] += KVx_0[n];
			KVx[index[n*3+1]-1] += KVx_1[n];
			KVx[index[n*3+2]-1] += KVx_2[n];
			KVy[index[n*3+0]-1] += KVy_0[n];
			KVy[index[n*3+1]-1] += KVy_1[n];
			KVy[index[n*3+2]-1] += KVy_2[n]; 

			/*Add basal friction*/
			if(groundedratio[n]>0.){
				for(int k=0;k<3;k++){
					for(int i=0;i<3;i++){
						for(int j=0;j<3;j++){
							if(i==j && j==k){
								KVx[index[n*3+k]-1] += groundedratio[n]*alpha2[index[n*3+i]-1]*vx[index[n*3+j]-1]*areas[n]/10.;
								KVy[index[n*3+k]-1] += groundedratio[n]*alpha2[index[n*3+i]-1]*vy[index[n*3+j]-1]*areas[n]/10.;
							}
							else if((i!=j) && (j!=k) && (k!=i)){
								KVx[index[n*3+k]-1] += groundedratio[n]*alpha2[index[n*3+i]-1]*vx[index[n*3+j]-1]*areas[n]/60.;
								KVy[index[n*3+k]-1] += groundedratio[n]*alpha2[index[n*3+i]-1]*vy[index[n*3+j]-1]*areas[n]/60.;
							}
							else{
								KVx[index[n*3+k]-1] += groundedratio[n]*alpha2[index[n*3+i]-1]*vx[index[n*3+j]-1]*areas[n]/30.;
								KVy[index[n*3+k]-1] += groundedratio[n]*alpha2[index[n*3+i]-1]*vy[index[n*3+j]-1]*areas[n]/30.;
							}
						}
					}
				}
			}        

		}    

	}
	__syncthreads();

}/*}}}*/
__global__ void  Main(double* ML, double* KVx, double* KVy, double* Fvx, double* Fvy, double* dVxdt, double* dVydt, double* vx, double* vy, double* resolx, double* resoly, double* H, double* eta_nbv, double* spcvx, double* spcvy, double eta_b, double rho, double damp, int nbv, int nbe) {/*{{{*/

	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	double ResVx;
	double ResVy;
	double dtVx;
	double dtVy;

	if(ix<nbv){
		/*1. Get time derivative based on residual (dV/dt)*/
		ResVx =  1./(rho*ML[ix])*(-KVx[ix] + Fvx[ix]); //rate of velocity in the x, equation 23
		ResVy =  1./(rho*ML[ix])*(-KVy[ix] + Fvy[ix]); //rate of velocity in the y, equation 24
		dVxdt[ix] = dVxdt[ix]*(1.-damp/20.) + ResVx;
		dVydt[ix] = dVydt[ix]*(1.-damp/20.) + ResVy;

		/*2. Explicit CFL time step for viscous flow, x and y directions*/
		dtVx = rho*pow(resolx[ix],2)/(4*H[ix]*eta_nbv[ix]*(1.+eta_b)*4.1);
		dtVy = rho*pow(resoly[ix],2)/(4*H[ix]*eta_nbv[ix]*(1.+eta_b)*4.1);

		/*3. velocity update, vx(new) = vx(old) + change in vx, Similarly for vy*/
		vx[ix] = vx[ix] + dVxdt[ix]*dtVx;
		vy[ix] = vy[ix] + dVydt[ix]*dtVy;

		/*Apply Dirichlet boundary condition*/
		if(!isnan(spcvx[ix])){
			vx[ix]    = 0.;
			dVxdt[ix] = 0.;
      }

		if(!isnan(spcvy[ix])){
			vy[ix]    = 0.;
			dVydt[ix] = 0.;
      }
	}

}/*}}}*/
__global__ void Viscosity(double* dvxdx, double* dvydy, double* dvxdy, double* dvydx, double* etan, double* rheology_B, double rele, double eta_0, double n_glen, int nbe){/*{{{*/

	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	double eps_xx;
	double eps_yy;
	double eps_xy;
	double EII2;
	double eta_it;
	double B;

	if(ix < nbe){
		eps_xx = dvxdx[ix];
		eps_yy = dvydy[ix];
		eps_xy = .5*(dvxdy[ix]+dvydx[ix]);
		B = rheology_B[ix];
		EII2 = pow(eps_xx,2) + pow(eps_yy,2) + pow(eps_xy,2) + eps_xx*eps_yy; 
		eta_it = 1.e+14/2.;  
		if(EII2>0.) eta_it = B/(2*pow(EII2,(n_glen-1.)/(2*n_glen)));

      etan[ix]  = min(exp(rele*log(eta_it) + (1-rele)*log(etan[ix])),eta_0*1e5);
   }

}/*}}}*/

/*Main*/
int main(){/*{{{*/

	/*Open input binary file*/
	const char* inputfile  = "./input.bin";
	const char* outputfile = "./output.outbin";
	FILE* fid = fopen(inputfile,"rb");
	if(fid==NULL) std::cerr<<"could not open file " << inputfile << " for binary reading or writing";

	/*Get All we need from binary file*/
	int    nbe,nbv,M,N;
	double g,rho,rho_w,yts;
	int    *index           = NULL;
	double *spcvx           = NULL;
	double *spcvy           = NULL;
	double *x               = NULL;
	double *y               = NULL;
	double *H               = NULL;
	double *surface         = NULL;
	double *base            = NULL;
	double *ice_levelset    = NULL;
	double *ocean_levelset  = NULL;
	double *rheology_B_temp = NULL;
	double *vx              = NULL;
	double *vy              = NULL;
	double *friction        = NULL;
	FetchData(fid,&nbe,"md.mesh.numberofelements");
	FetchData(fid,&nbv,"md.mesh.numberofvertices");
	FetchData(fid,&g,"md.constants.g");
	FetchData(fid,&rho,"md.materials.rho_ice");
	FetchData(fid,&rho_w,"md.materials.rho_water");
	FetchData(fid,&yts,"md.constants.yts");
	FetchData(fid,&index,&M,&N,"md.mesh.elements");
	FetchData(fid,&spcvx,&M,&N,"md.stressbalance.spcvx");
	FetchData(fid,&spcvy,&M,&N,"md.stressbalance.spcvy");
	FetchData(fid,&x,&M,&N,"md.mesh.x");
	FetchData(fid,&y,&M,&N,"md.mesh.y");
	FetchData(fid,&H,&M,&N,"md.geometry.thickness");
	FetchData(fid,&surface,&M,&N,"md.geometry.surface");
	FetchData(fid,&base,&M,&N,"md.geometry.base");
	FetchData(fid,&ice_levelset,&M,&N,"md.mask.ice_levelset");
	FetchData(fid,&ocean_levelset,&M,&N,"md.mask.ocean_levelset");
	FetchData(fid,&rheology_B_temp,&M,&N,"md.materials.rheology_B");
	FetchData(fid,&vx,&M,&N,"md.initialization.vx");
	FetchData(fid,&vy,&M,&N,"md.initialization.vy");
	FetchData(fid,&friction,&M,&N,"md.friction.coefficient");

	/*Close input file*/
	if(fclose(fid)!=0) std::cerr<<"could not close file " << inputfile;

	/*Constants*/
	double n_glen    = 3.;
	double damp      = 2.;
	double rele      = 1e-1;
	double eta_b     = 0.5;
	double eta_0     = 1.e+14/2.;
	int    niter     = 5e6;
	int    nout_iter = 1000;
	double epsi      = 1e-8;

	/*Initial guesses (except vx and vy that we already loaded)*/
	double* etan = new double[nbe];
	for(int i=0;i<nbe;i++) etan[i] = 1.e+14;
	double* dVxdt = new double[nbv];
	for(int i=0;i<nbv;i++) dVxdt[i] = 0.;
	double* dVydt = new double[nbv];
	for(int i=0;i<nbv;i++) dVydt[i] = 0.;

	/*Manage derivatives once for all*/
	double* alpha   = NULL;
	double* beta    = NULL;
	double* areas   = NULL;
	double* weights = NULL;
	NodalCoeffs(&areas,&alpha,&beta,index,x,y,nbe);
	Weights(&weights,index,areas,nbe,nbv);

	/*MeshSize*/
	double* resolx = new double[nbv];
	double* resoly = new double[nbv];
	MeshSize(resolx,resoly,index,x,y,areas,weights,nbe,nbv);

	/*Physical properties once for all*/
	double* dsdx = new double[nbe];
	double* dsdy = new double[nbe];
	derive_xy_elem(dsdx,dsdy,surface,index,alpha,beta,nbe);
	double* Helem      = new double[nbe];
	double* rheology_B = new double[nbe];
	for(int i=0;i<nbe;i++){
		Helem[i]      = 1./3. * (H[index[i*3+0]-1] + H[index[i*3+1]-1] + H[index[i*3+2]-1]);
		rheology_B[i] = 1./3. * (rheology_B_temp[index[i*3+0]-1] + rheology_B_temp[index[i*3+1]-1] + rheology_B_temp[index[i*3+2]-1]);
	}

	/*Compute RHS amd ML once for all*/
	double* ML            = new double[nbv];
	double* Fvx           = new double[nbv];
	double* Fvy           = new double[nbv];
	double* groundedratio = new double[nbe];
	for(int i=0;i<nbv;i++){
		ML[i]  = 0.;
		Fvx[i] = 0.;
		Fvy[i] = 0.;
	}
	for(int n=0;n<nbe;n++){
		/*Lumped mass matrix*/
		for(int i=0;i<3;i++){
			for(int j=0;j<3;j++){
				// \int_E phi_i * phi_i dE = A/6 and % \int_E phi_i * phi_j dE = A/12
				if(i==j)
				 ML[index[n*3+j]-1] += areas[n]/6.;
				else
				 ML[index[n*3+j]-1] += areas[n]/12.;
			}
		}
		/*RHS, 'F ' in equation 22 (Driving Stress)*/
		for(int i=0;i<3;i++){
			for(int j=0;j<3;j++){
				if(i==j){
					Fvx[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdx[n]*areas[n]/6.;
					Fvy[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdy[n]*areas[n]/6.;
				}
				else{
					Fvx[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdx[n]*areas[n]/12.;
					Fvy[index[n*3+i]-1] += -rho*g*H[index[n*3+j]-1]*dsdy[n]*areas[n]/12.;
				}
			}
		}
	}

	/*RHS (Water pressure at the ice front)*/
	double level[3];
	for(int n=0;n<nbe;n++){
		/*Determine if there is an ice front there*/
		level[0] = ice_levelset[index[n*3+0]-1];
		level[1] = ice_levelset[index[n*3+1]-1];
		level[2] = ice_levelset[index[n*3+2]-1];
		int count = 0;
		for(int i=0;i<3;i++) if(level[i]<0.) count++;
		if(count==1){

			/*Ok this element has an ice front, get indices of the 2 vertices*/
			int seg1[2] = {index[n*3+0]-1,index[n*3+1]-1};
			int seg2[2] = {index[n*3+1]-1,index[n*3+2]-1};
			int seg3[2] = {index[n*3+2]-1,index[n*3+0]-1};
			int pairids[2];
			if(ice_levelset[seg1[0]]>=0 && ice_levelset[seg1[1]]>=0){
				pairids[0] = seg1[0]; pairids[1] = seg1[1];
			}
			else if (ice_levelset[seg2[0]]>=0 && ice_levelset[seg2[1]]>=0){
				pairids[0] = seg2[0]; pairids[1] = seg2[1];
			}
			else if (ice_levelset[seg3[0]]>=0 && ice_levelset[seg3[1]]>=0){
				pairids[0] = seg3[0]; pairids[1] = seg3[1];
			}
			else{
				std::cerr<<"case not supported";
			}

			/*Get normal*/
			double len = sqrt(pow(x[pairids[1]]-x[pairids[0]],2) + pow(y[pairids[1]]-y[pairids[0]],2) );
			double nx  = +(y[pairids[1]]-y[pairids[0]])/len;
			double ny  = -(x[pairids[1]]-x[pairids[0]])/len;

			/*RHS*/
			for(int i=0;i<2;i++){
				for(int j=0;j<2;j++){
					double bibj = base[pairids[i]]*base[pairids[j]];
					double HiHj = H[pairids[i]]*H[pairids[j]];
					for(int k=0;k<2;k++){
						if(i==j && j==k){
							Fvx[pairids[k]] += 1./2.*(-rho_w*g*bibj+rho*g*HiHj)*nx*len/4.;
							Fvy[pairids[k]] += 1./2.*(-rho_w*g*bibj+rho*g*HiHj)*ny*len/4.;
						}
						else{
							Fvx[pairids[k]] += 1./2.*(-rho_w*g*bibj+rho*g*HiHj)*nx*len/12.;
							Fvy[pairids[k]] += 1./2.*(-rho_w*g*bibj+rho*g*HiHj)*ny*len/12.;
						}
					}
				}
			}
		}

		/*One more thing in this element loop: prepare groundedarea needed later for the calculation of basal friction*/
		level[0] = ocean_levelset[index[n*3+0]-1];
		level[1] = ocean_levelset[index[n*3+1]-1];
		level[2] = ocean_levelset[index[n*3+2]-1];
		if(level[0]>0. && level[1]>0. && level[2]>0.){
			/*Completely grounded*/
			groundedratio[n]=1.;
		}
		else if(level[0]<0. && level[1]<0. && level[2]<0.){
			/*Completely floating*/
			groundedratio[n]=0.;
		}
		else{
			/*Partially floating,*/
			double s1,s2;
			if(level[0]*level[1]>0){/*Nodes 0 and 1 are similar, so points must be found on segment 0-2 and 1-2*/
				s1=level[2]/(level[2]-level[1]);
				s2=level[2]/(level[2]-level[0]);
			}
			else if(level[1]*level[2]>0){ /*Nodes 1 and 2 are similar, so points must be found on segment 0-1 and 0-2*/
				s1=level[0]/(level[0]-level[1]);
				s2=level[0]/(level[0]-level[2]);
			}
			else if(level[0]*level[2]>0){/*Nodes 0 and 2 are similar, so points must be found on segment 1-0 and 1-2*/
				s1=level[1]/(level[1]-level[0]);
				s2=level[1]/(level[1]-level[2]);
			}
			else{
				std::cerr<<"should not be here...";
			}

			if(level[0]*level[1]*level[2]>0.){
				/*two nodes floating, inner triangle is grounded*/
				groundedratio[n]= s1*s2;
			}
			else{
				/*one node floating, inner triangle is floating*/
				groundedratio[n]= (1.-s1*s2);
			}
		}
	}

	/*Finally add calculation of friction coefficient*/
	double* alpha2 = new double[nbv];
	for(int i=0;i<nbv;i++){
		/*Compute effective pressure*/
		double p_ice   = g*rho*H[i];
		double p_water = -rho_w*g*base[i];
		double Neff    = p_ice - p_water;
		if(Neff<0.) Neff=0.;

		/*Compute alpha2*/
		alpha2[i] = pow(friction[i],2)*Neff;
	}

   /*------------ now copy all relevant vectors from host to device ---------------*/

	int *d_index = NULL;
	cudaMalloc(&d_index, nbe*3*sizeof(int));
	cudaMemcpy(d_index, index, nbe*3*sizeof(int), cudaMemcpyHostToDevice);

	double *d_vx;
	cudaMalloc(&d_vx, nbv*sizeof(double));
	cudaMemcpy(d_vx, vx, nbv*sizeof(double), cudaMemcpyHostToDevice);  

	double *d_vy;
	cudaMalloc(&d_vy, nbv*sizeof(double));
	cudaMemcpy(d_vy, vy, nbv*sizeof(double), cudaMemcpyHostToDevice);  

	double *d_alpha;
	cudaMalloc(&d_alpha, nbe*3*sizeof(double));
	cudaMemcpy(d_alpha, alpha, nbe*3*sizeof(double), cudaMemcpyHostToDevice);

	double *d_beta;
	cudaMalloc(&d_beta, nbe*3*sizeof(double));
	cudaMemcpy(d_beta, beta, nbe*3*sizeof(double), cudaMemcpyHostToDevice);

	double *d_etan;
	cudaMalloc(&d_etan, nbe*sizeof(double));
	cudaMemcpy(d_etan, etan, nbe*sizeof(double), cudaMemcpyHostToDevice);  

	double *d_rheology_B;
	cudaMalloc(&d_rheology_B, nbe*sizeof(double));
	cudaMemcpy(d_rheology_B, rheology_B, nbe*sizeof(double), cudaMemcpyHostToDevice); 

	double *d_Helem;
	cudaMalloc(&d_Helem, nbe*sizeof(double));
	cudaMemcpy(d_Helem, Helem, nbe*sizeof(double), cudaMemcpyHostToDevice); 

	double *d_areas;
	cudaMalloc(&d_areas, nbe*sizeof(double));
	cudaMemcpy(d_areas, areas, nbe*sizeof(double), cudaMemcpyHostToDevice); 

   double* eta_nbv = new double[nbv];
	double *d_eta_nbv;
	cudaMalloc(&d_eta_nbv, nbv*sizeof(double));
	cudaMemcpy(d_eta_nbv, eta_nbv, nbv*sizeof(double), cudaMemcpyHostToDevice);/*FIXME: do we REALLY need to copy?? it's empty right now*/

	double *d_weights;
	cudaMalloc(&d_weights, nbv*sizeof(double));
	cudaMemcpy(d_weights, weights, nbv*sizeof(double), cudaMemcpyHostToDevice);  

	double *d_ML;
	cudaMalloc(&d_ML, nbv*sizeof(double));
	cudaMemcpy(d_ML, ML, nbv*sizeof(double), cudaMemcpyHostToDevice);  

	double *d_Fvx;
	cudaMalloc(&d_Fvx, nbv*sizeof(double));
	cudaMemcpy(d_Fvx, Fvx, nbv*sizeof(double), cudaMemcpyHostToDevice); 

	double *d_Fvy;
	cudaMalloc(&d_Fvy, nbv*sizeof(double));
	cudaMemcpy(d_Fvy, Fvy, nbv*sizeof(double), cudaMemcpyHostToDevice); 

	double *d_dVxdt;
	cudaMalloc(&d_dVxdt, nbv*sizeof(double));
	cudaMemcpy(d_dVxdt, dVxdt, nbv*sizeof(double), cudaMemcpyHostToDevice); 

	double *d_dVydt;
	cudaMalloc(&d_dVydt, nbv*sizeof(double));
	cudaMemcpy(d_dVydt, dVydt, nbv*sizeof(double), cudaMemcpyHostToDevice); 

	double *d_resolx;
	cudaMalloc(&d_resolx, nbv*sizeof(double));
	cudaMemcpy(d_resolx, resolx, nbv*sizeof(double), cudaMemcpyHostToDevice);

	double *d_resoly;
	cudaMalloc(&d_resoly, nbv*sizeof(double));
	cudaMemcpy(d_resoly, resoly, nbv*sizeof(double), cudaMemcpyHostToDevice);

	double *d_H;
	cudaMalloc(&d_H, nbv*sizeof(double));
	cudaMemcpy(d_H, H, nbv*sizeof(double), cudaMemcpyHostToDevice);

	double *d_spcvx;
	cudaMalloc(&d_spcvx, nbv*sizeof(double));
	cudaMemcpy(d_spcvx, spcvx, nbv*sizeof(double), cudaMemcpyHostToDevice);

	double *d_spcvy;
	cudaMalloc(&d_spcvy, nbv*sizeof(double));
	cudaMemcpy(d_spcvy, spcvy, nbv*sizeof(double), cudaMemcpyHostToDevice);

	double *d_alpha2;
	cudaMalloc(&d_alpha2, nbv*sizeof(double));
	cudaMemcpy(d_alpha2, alpha2, nbv*sizeof(double), cudaMemcpyHostToDevice);

	double *d_groundedratio;
	cudaMalloc(&d_groundedratio, nbe*sizeof(double));
	cudaMemcpy(d_groundedratio, groundedratio, nbe*sizeof(double), cudaMemcpyHostToDevice);


   /*------------ allocate relevant vectors on host (GPU)---------------*/

	double *dvxdx = new double[nbe]; /*FIXME: I don't think you need to allocate on host*/
	cudaMalloc(&dvxdx,nbe*sizeof(double));

	double *dvxdy =  new double[nbe];
	cudaMalloc(&dvxdy, nbe*sizeof(double));

	double *dvydx = new double[nbe];
	cudaMalloc(&dvydx, nbe*sizeof(double));

	double *dvydy = new double[nbe];
	cudaMalloc(&dvydy, nbe*sizeof(double));

	double *KVx = new double[nbv];
	cudaMalloc(&KVx, nbv*sizeof(double));

	double *KVy = new double[nbv];
	cudaMalloc(&KVy, nbv*sizeof(double));

	double *KVx_0 = new double[nbe];
	cudaMalloc(&KVx_0, nbe*sizeof(double));

	double *KVx_1 = new double[nbe];
	cudaMalloc(&KVx_1, nbe*sizeof(double));

	double *KVx_2 = new double[nbe];
	cudaMalloc(&KVx_2, nbe*sizeof(double));

	double *KVy_0 = new double[nbe];
	cudaMalloc(&KVy_0, nbe*sizeof(double));

	double *KVy_1 = new double[nbe];
	cudaMalloc(&KVy_1, nbe*sizeof(double));

	double *KVy_2 = new double[nbe];
	cudaMalloc(&KVy_2, nbe*sizeof(double));

	double *KVxx = new double[nbe];
	cudaMalloc(&KVxx, nbe*sizeof(double));

	double *KVyy = new double[nbe];
	cudaMalloc(&KVyy, nbe*sizeof(double));

	double *eta_nbv_gpu = new double[nbe];
	cudaMalloc(&eta_nbv_gpu, nbe*sizeof(double));

	/*Main loop*/
	double normX; /*FIXME: do we need to declare these variables here?*/
	double normY;
	double iterror;
	for(int iter=1;iter<=niter;iter++){

		/*'KV' term in equation 22 - GPU KERNEL 1*/ 
		KV <<<gridSize,blockSize>>> (KVx, KVy, KVx_0, KVx_1, KVx_2, KVy_0, KVy_1, KVy_2, KVxx, KVyy, d_vx, d_vy, d_alpha2, d_groundedratio, d_etan, dvxdx, dvydy, dvxdy, dvydx, d_Helem, d_areas, d_alpha, d_beta, d_index, nbv, nbe, d_eta_nbv, eta_nbv_gpu, d_weights); cudaDeviceSynchronize();               

		/*Velocity rate update in the x and y, refer to equation 19 in Rass paper*/
		normX = 0.;
		normY = 0.;

		/*dVxdt, dVydt - GPU KERNEL 2*/
		Main <<<gridSize,blockSize>>> (d_ML, KVx, KVy, d_Fvx, d_Fvy, d_dVxdt, d_dVydt, d_vx, d_vy, d_resolx, d_resoly, d_H, d_eta_nbv, d_spcvx, d_spcvy, eta_b, rho, damp, nbv, nbe); cudaDeviceSynchronize();  

      /*4. Update error*/
      /*FIXME: can we do this in Main so that we don't need to copy the WHOLE vector from device to host?*/
		cudaMemcpy(dVxdt, d_dVxdt, nbv*sizeof(double), cudaMemcpyDeviceToHost );
		cudaMemcpy(dVydt, d_dVydt, nbv*sizeof(double), cudaMemcpyDeviceToHost );
		for(int i=0;i<nbv;i++){
			normX += pow(dVxdt[i],2);
			normY += pow(dVydt[i],2);
		}

		/*Get final error estimate*/
		normX  = sqrt(normX)/double(nbv);
		normY  = sqrt(normY)/double(nbv);

		/*Check convergence*/
		iterror = max(normX,normY);
		if((iterror < epsi) && (iter > 2)) break;
		if ((iter%nout_iter)==1){
			std::cout<<"iter="<<iter<<", err="<<iterror<<std::endl;
		}

		/*Updayte Viscosity - GPU KERNEL 3*/ /*FIXME: why not do this within the second Kernel if that saves time?*/
		Viscosity <<<gridSize,blockSize>>>  (dvxdx, dvydy, dvxdy, dvydx, d_etan, d_rheology_B, rele, eta_0, n_glen, nbe);  cudaDeviceSynchronize();  
	}

   /*Copy results from Device to host*/
	cudaMemcpy(vx, d_vx, nbv*sizeof(double), cudaMemcpyDeviceToHost );
	cudaMemcpy(vy, d_vy, nbv*sizeof(double), cudaMemcpyDeviceToHost ); 
   std::cout<<"iter="<<iter<<", err="<<iterror<<std::endl;

	/*Write output*/
	fid = fopen(outputfile,"wb");
	if(fid==NULL) std::cerr<<"could not open file " << outputfile << " for binary reading or writing";
	WriteData(fid,"PTsolution","SolutionType");
	WriteData(fid,vx,nbv,1,"Vx");
	WriteData(fid,vy,nbv,1,"Vy");
	if(fclose(fid)!=0) std::cerr<<"could not close file " << outputfile;

	/*Cleanup and return*/
   delete [] eta_nbv;
	delete [] index;
	delete [] x;
	delete [] y;
	delete [] H;
	delete [] surface;
	delete [] base;
	delete [] spcvx;
	delete [] spcvy;
	delete [] ice_levelset;
	delete [] ocean_levelset;
	delete [] rheology_B;
	delete [] rheology_B_temp;
	delete [] vx;
	delete [] vy;
	delete [] friction;
	delete [] alpha2;
	delete [] etan;
	delete [] dVxdt;
	delete [] dVydt;
	delete [] alpha;
	delete [] beta;
	delete [] areas;
	delete [] weights;
	delete [] resolx;
	delete [] resoly;
	delete [] dsdx;
	delete [] dsdy;
	delete [] Helem;
	delete [] ML;
	delete [] Fvx;
	delete [] Fvy;

	cudaFree(d_vx);
	cudaFree(d_vy);
	cudaFree(d_index);
	cudaFree(d_alpha);
	cudaFree(d_beta);
	cudaFree(d_weights);
	cudaFree(d_areas);
	cudaFree(dvxdx);
	cudaFree(dvxdy);
	cudaFree(dvydx);
	cudaFree(dvydy);
	cudaFree(KVx);
	cudaFree(KVy);
	cudaFree(KVx_0);
	cudaFree(KVx_1);
	cudaFree(KVx_2);
	cudaFree(KVy_0);
	cudaFree(KVy_1);
	cudaFree(KVy_2);
	cudaFree(KVxx);
	cudaFree(KVyy);
	cudaFree(eta_nbv_gpu);
	cudaFree(eta_nbv);
	cudaFree(d_etan);
	cudaFree(d_ML);
	cudaFree(d_Fvx);
	cudaFree(d_Fvy);
	cudaFree(d_dVxdt);
	cudaFree(d_dVydt);
	cudaFree(d_resolx);
	cudaFree(d_resoly);
	cudaFree(d_H);
	cudaFree(d_Helem);
	cudaFree(d_spcvx);
	cudaFree(d_spcvy);   
	cudaFree(d_alpha2);
	cudaFree(d_groundedratio);

	return 0;
}/*}}}*/

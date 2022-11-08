/*CPU Code*/
/*I/O stuff*/
typedef double ftype;
FILE* SetFilePointerToData(FILE* fid,int* pcode,int* pvector_type,const char* data_name){

    int found  = 0;
    const char* mddot = "md.";
    char* record_name = NULL;
    int   record_name_size;
    long long record_length;
    int record_code;       //1 to 7 number
    int vector_type   = 0; //nodal or elementary

    if(strncmp(data_name,mddot,3)!=0){
        std::cerr <<"Cannot fetch \"JKS_SingleRun_rev.cu"<<data_name<<"\" does not start with \""<<mddot<<"\"";
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

void FetchData(FILE* fid,int* pinteger,const char* data_name){

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
}

void FetchData(FILE* fid,int** pmatrix,int* pM,int* pN,const char* data_name){

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
}

void FetchData(FILE* fid,ftype* pdouble,const char* data_name){

    /*output: */
    ftype value;
    double tempValue;
    int code;

    /*Set file pointer to beginning of the data: */
    fid=SetFilePointerToData(fid,&code,NULL,data_name);

    if(code!=3)std::cerr <<"expecting a ftype for \"" << data_name<<"\"";

    /*We have to read a integer from disk. First read the dimensions of the integer, then the integer: */
    if(fread(&tempValue,sizeof(double),1,fid)!=1) std::cerr<<"could not read scalar";

    value = ftype(tempValue);

    /*Assign output pointers: */
    *pdouble=value;
}

void FetchData(FILE* fid,ftype** pmatrix,int* pM,int* pN,const char* data_name){

    /*output: */
    int M,N;
    ftype* matrix=NULL;
    // int* integer_matrix=NULL;
    int code=0;

    /*Set file pointer to beginning of the data: */
    fid=SetFilePointerToData(fid,&code,NULL,data_name);
    if(code!=5 && code!=6 && code!=7)std::cerr<<"expecting a Issmftype, integer or boolean matrix for \""<<data_name<<"\""<<" (Code is "<<code<<")";

    /*Now fetch: */

    /*We have to read a matrix from disk. First read the dimensions of the matrix, then the whole matrix: */
    /*numberofelements: */
    if(fread(&M,sizeof(int),1,fid)!=1) std::cerr<<"could not read number of rows for matrix ";
    if(fread(&N,sizeof(int),1,fid)!=1) std::cerr<<"could not read number of columns for matrix ";

    /*Now allocate matrix: */
    if(M*N){
        double* tempMatrix=new double[M*N];

        /*Read matrix on node 0, then broadcast: */
        if(fread(tempMatrix,M*N*sizeof(double),1,fid)!=1) std::cerr<<"could not read matrix ";
        matrix=new ftype[M*N];

        for (int s=0; s<M*N; s++)
            matrix[s] = ftype(tempMatrix[s]);
        delete[] tempMatrix;        
    }

    /*Assign output pointers: */
    *pmatrix=matrix;
    if(pM)*pM=M;
    if(pN)*pN=N;
}

void WriteData(FILE* fid,ftype* matrix,int M,int N,const char* data_name){

    /*First write enum: */
    int length=(strlen(data_name)+1)*sizeof(char);
    fwrite(&length,sizeof(int),1,fid);
    fwrite(data_name,length,1,fid);

    /*Now write time and step: */
    ftype time = 0.;
    int    step = 1;
    fwrite(&time,sizeof(ftype),1,fid);
    fwrite(&step,sizeof(int),1,fid);

    /*writing a IssmDouble array, type is 3:*/
    int type=3;
    fwrite(&type,sizeof(int),1,fid);
    fwrite(&M,sizeof(int),1,fid);
    fwrite(&N,sizeof(int),1,fid);
    fwrite(matrix,M*N*sizeof(ftype),1,fid);
}

void WriteData(FILE* fid,const char* string,const char* data_name){

    /*First write enum: */
    int length=(strlen(data_name)+1)*sizeof(char);
    fwrite(&length,sizeof(int),1,fid);
    fwrite(data_name,length,1,fid);

    /*Now write time and step: */
    ftype time = 0.;
    int    step = 1;
    fwrite(&time,sizeof(ftype),1,fid);
    fwrite(&step,sizeof(int),1,fid);

    /*writing a string, type is 2: */
    int type=2;
    fwrite(&type,sizeof(int),1,fid);

    length=(strlen(string)+1)*sizeof(char);
    fwrite(&length,sizeof(int),1,fid);
    fwrite(string,length,1,fid);
}

void NodalCoeffs(ftype** pareas,ftype** palpha,ftype** pbeta,int* index,ftype* x,ftype* y,int nbe){

    /*Allocate output vectors*/
    ftype* areas = new ftype[nbe];
    ftype* alpha = new ftype[nbe*3];
    ftype* beta  = new ftype[nbe*3];

    /*Loop over all elements and calculate nodal function coefficients and element surface area*/
    for(int i = 0; i < nbe; i++) {
        int n1 = index[i*3+0]-1;
        int n2 = index[i*3+1]-1;
        int n3 = index[i*3+2]-1;

        ftype x1 = x[n1];
        ftype x2 = x[n2];
        ftype x3 = x[n3];
        ftype y1 = y[n1];
        ftype y2 = y[n2];
        ftype y3 = y[n3];

        ftype invdet = 1./(x1 * (y2 - y3) - x2 * (y1 - y3) + x3 * (y1 - y2));

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
}

void Weights(ftype** pweights,int* index,ftype* areas,int nbe,int nbv){

    /*Allocate output and initialize as 0*/
    ftype* weights = new ftype[nbv];
    for(int i = 0; i < nbv; i++) weights[i]=0.;

    /*Loop over elements*/
    for(int i = 0; i < nbe; i++){
        for(int j = 0; j < 3; j++){
            weights[index[i*3+j]-1] += areas[i];
        }
    }

    /*Assign output pointer*/
    *pweights = weights;
}

void derive_xy_elem(ftype* dfdx_e,ftype* dfdy_e,ftype* f,int* index,ftype* alpha,ftype* beta,int nbe){

    /*WARNING!! Assume that dfdx_e and dfdy_e have been properly allocated*/

    for(int i=0;i<nbe;i++){
        int n1 = index[i*3+0]-1;
        int n2 = index[i*3+1]-1;
        int n3 = index[i*3+2]-1;
        dfdx_e[i] = f[n1]*alpha[i*3+0] + f[n2]*alpha[i*3+1] + f[n3]*alpha[i*3+2];
        dfdy_e[i] = f[n1]*beta[ i*3+0] + f[n2]*beta[ i*3+1] + f[n3]*beta[ i*3+2];
    }
}

void elem2node(ftype* f_v,ftype* f_e,int* index,ftype* areas,ftype* weights,int nbe,int nbv){

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
}

void MeshSize(ftype* resolx,ftype* resoly,int* index,ftype* x,ftype* y,ftype* areas,ftype* weights,int nbe,int nbv){

    /*Get element size along x and y directions*/
    ftype  xmin,xmax,ymin,ymax;
    ftype* dx_elem = new ftype[nbe];
    ftype* dy_elem = new ftype[nbe];
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
}


////////////////////////
////////////// GPU stuff
////////////////////////

void clean_cuda(){ 
    cudaError_t ce = cudaGetLastError();
    if(ce != cudaSuccess){ printf("ERROR launching GPU C-CUDA program: %s\n", cudaGetErrorString(ce)); cudaDeviceReset(); }
}

// Find the max of an array
__shared__ volatile ftype block_maxval;
__global__ void __device_max_d(ftype* A, int nbv, ftype* device_maxval){
   
    ftype thread_maxval=0.0;
   
    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    // find the maxval for each block
    if (ix<nbv){ thread_maxval = abs(A[ix]); }
    if (threadIdx.x==0){ block_maxval=0.0; }
    __syncthreads();
    for (int i=0; i < (BLOCK_Xv); i++){
           if (i==threadIdx.x) if(isnan(thread_maxval)) {{block_maxval = thread_maxval;} break;} 
           else { block_maxval = max(block_maxval, thread_maxval);}  
   //    if (i==threadIdx.x){ block_maxval = max(block_maxval, thread_maxval); }
        __syncthreads();
    }
    device_maxval[blockIdx.x] = block_maxval;
//    std::cerr<<"Found NaN in dVydt[i]";
//    printf("Found NaN in dVxdt[i] \n");
}

#define __device_max_x(dVxdt)   __device_max_d<<<gridv, blockv>>>(d_dVxdt, nbv, d_device_maxvalx); \
                                cudaMemcpy(device_maxvalx, d_device_maxvalx, GRID_Xv*sizeof(ftype), cudaMemcpyDeviceToHost); \
                                ftype device_MAXx = 0.0;                                     \
                                for (int i=0; i < (GRID_Xv); i++){                            \
                                     if(isnan(device_maxvalx[i])) {device_MAXx = device_maxvalx[i];std::cerr<<"\n Found NaN in dVxdt[i]"; break;} \
                                     device_MAXx = max(device_MAXx, device_maxvalx[i]);        \
                                }

#define __device_max_y(dVydt)   __device_max_d<<<gridv, blockv>>>(d_dVydt, nbv, d_device_maxvaly); \
                                cudaMemcpy(device_maxvaly, d_device_maxvaly, GRID_Xv*sizeof(ftype), cudaMemcpyDeviceToHost); \
                                ftype device_MAXy = 0.0;                                     \
                                for (int i=0; i < (GRID_Xv); i++){                            \
                                         if(isnan(device_maxvaly[i])) {device_MAXy = device_maxvaly[i]; std::cerr<<"\n Found NaN in dVydt[i]"; break;} \
                                         device_MAXy = max(device_MAXy, device_maxvaly[i]);        \
                                }

// timer
#include "sys/time.h"
ftype timer_start = 0;
ftype cpu_sec(){ struct timeval tp; gettimeofday(&tp,NULL); return tp.tv_sec+1e-6*tp.tv_usec; }
void   tic(){ timer_start = cpu_sec(); }
ftype toc(){ return cpu_sec()-timer_start; }
void   tim(const char *what, ftype n){ ftype s=toc(); printf("%s: %8.3f seconds",what,s);if(n>0)printf(", %8.3f GB/s", n/s); printf("\n"); }
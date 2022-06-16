#include <cstdio>
#include <iostream>
#include <string.h>
#include <cmath>
using namespace std;

/*define GPU specific variables*/
#define GPU_ID    6

#define BLOCK_Xe  128
#define BLOCK_Xv  128

// Device norm subroutine
#define blockId       (blockIdx.x)
#define threadId      (threadIdx.x)
#define isBlockMaster (threadIdx.x==0)

#include "helpers.h"

/*CUDA Code*/
// __global__ void PT1(double* Tmp, double* alpha, int nbe){ 
__global__ void PT1(double* Tmp, double* alpha_0, double* alpha_1, double* alpha_2, int nbe){ 
    for(int ix = blockIdx.x * blockDim.x + threadIdx.x; ix<nbe; ix += blockDim.x * gridDim.x){
    // int ix = blockIdx.x * blockDim.x + threadIdx.x; if (ix<nbe){
    // double dvxdx = alpha[ix*3+0] + alpha[ix*3+1] + alpha[ix*3+2];  //column wise
    // double dvxdx = alpha_0[ix] + alpha_1[ix]  +  alpha_2[ix];  //separate arrays
    // double dvxdx = alpha[ix] + alpha[ix +nbe*1] + alpha[ix + nbe*2];   //row wise 
    // Tmp[ix] = alpha[ix*3+0] + alpha[ix*3+1] + alpha[ix*3+2];  //column wise
    Tmp[ix] = alpha_0[ix] + alpha_1[ix]  +  alpha_2[ix];  //separate arrays
    // Tmp[ix] = alpha[ix] + alpha[ix +nbe*1] + alpha[ix + nbe*2];   //row wise    
    }
}

//Option suggested by Oded//

  // for(int ix = blockIdx.x * blockDim.x + threadIdx.x; ix<nbe*3; ix += blockDim.x * gridDim.x){ 
          
     
      //  double dvxdx = alpha[ix+0] + alpha[ix+1] + alpha[ix+2];  //column wise
      //setting dummy rows to zeroes, making longer would imply making index longer which is not possible.
                         
           
 //   }

/*Main*/
int main(){

    /*Open input binary file*/
    // const char* inputfile  = "../inputfiles/PIG3e4.bin";
    // const char* inputfile  = "../inputfiles/PIG8e6.bin";
    const char* inputfile  = "../inputfiles/JKS2e7.bin";
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
    double n_glen     = 3.;
    double damp       = 0.981; //0.96 for JKS2e4, 0.981 for PIG3e4
    double rele       = 0.07;   //1e-1 for JKS2e4, 0.07 for PIG3e4
    double eta_b      = 0.5;
    double eta_0      = 1.e+14/2.0;
    int    niter      = 10000; //5e6
    // int    nout_iter  = 1; //change it to 100 for JKS2e4
    double epsi       = 3.171e-7;
    double relaxation = 0.967; //0.7 for JKS2e4, 0.967 for PIG3e4
        
    // Ceiling division to get the close to optimal GRID size
    unsigned int GRID_Xe = 1 + ((nbe - 1) / BLOCK_Xe);
    unsigned int GRID_Xv = 1 + ((nbv - 1) / BLOCK_Xv);

    GRID_Xe = GRID_Xe - GRID_Xe%80;
    GRID_Xv = GRID_Xv - GRID_Xv%80;

    std::cout<<"GRID_Xe="<<GRID_Xe<<std::endl;
    std::cout<<"GRID_Xv="<<GRID_Xv<<std::endl;

    // Set up GPU
    int gpu_id=-1;
    dim3 gridv, blockv;
    dim3 gride, blocke;
    blockv.x = BLOCK_Xv; gridv.x = GRID_Xv;
    blocke.x = BLOCK_Xe; gride.x = GRID_Xe;
    gpu_id = GPU_ID; cudaSetDevice(gpu_id); cudaGetDevice(&gpu_id);
    cudaDeviceReset(); cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
    printf("Process uses GPU with id %d.\n", gpu_id);
    //cudaSetDevice  selects the device, set the gpu id you selected

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

////Separate arrays///
    double* Tmp       = new double[nbe];
    double* alpha_0   = new double[nbe];
    double* alpha_1   = new double[nbe];
    double* alpha_2   = new double[nbe];
    
      for(int i=0;i<nbe;i++){
           alpha_0[i] =  alpha[i*3];
           alpha_1[i] =  alpha[i*3+1];
           alpha_2[i] =  alpha[i*3+2];
        } 

///////////////////////////////////////

////To  rearrange  arrays in alpha -- row wise//


//  for(int j=0;j<3;j++){
//      for(int i=0;i<nbe;i++){
//         alpha[i + j*nbe] = alpha[i*3+j];
//     }
// }

//////////////////////////////////////////////////

    
    /*------------ now copy all relevant vectors from host to device ---------------*/

    double *d_vx;
    cudaMalloc(&d_vx, nbv*sizeof(double));
    cudaMemcpy(d_vx, vx, nbv*sizeof(double), cudaMemcpyHostToDevice);  

    double *d_vy;
    cudaMalloc(&d_vy, nbv*sizeof(double));
    cudaMemcpy(d_vy, vy, nbv*sizeof(double), cudaMemcpyHostToDevice);  

    double *d_alpha;
    cudaMalloc(&d_alpha, nbe*3*sizeof(double));
    cudaMemcpy(d_alpha, alpha, nbe*3*sizeof(double), cudaMemcpyHostToDevice);
    
    double *d_alpha_0;
    cudaMalloc(&d_alpha_0, nbe*sizeof(double));
    cudaMemcpy(d_alpha_0, alpha_0, nbe*sizeof(double), cudaMemcpyHostToDevice);

    double *d_alpha_1;
    cudaMalloc(&d_alpha_1, nbe*sizeof(double));
    cudaMemcpy(d_alpha_1, alpha_1, nbe*sizeof(double), cudaMemcpyHostToDevice);


    double *d_alpha_2;
    cudaMalloc(&d_alpha_2, nbe*sizeof(double));
    cudaMemcpy(d_alpha_2, alpha_2, nbe*sizeof(double), cudaMemcpyHostToDevice);

    double *d_Tmp;
    cudaMalloc(&d_Tmp, nbe*sizeof(double));
    cudaMemcpy(d_Tmp, Tmp, nbe*sizeof(double), cudaMemcpyHostToDevice);

    
    /*------------ allocate relevant vectors on host (GPU)---------------*/
    double *dvxdx = NULL;
    cudaMalloc(&dvxdx,nbe*sizeof(double));

    double *dvxdy = NULL;
    cudaMalloc(&dvxdy, nbe*sizeof(double));

    double *dvydx = NULL;
    cudaMalloc(&dvydx, nbe*sizeof(double));

    double *dvydy = NULL;
    cudaMalloc(&dvydy, nbe*sizeof(double));
    
    //Creating CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Perf
    double time_s = 0.0;
    double mem = (double)1e-9*(double)nbv*sizeof(double);
    // int nIO = 8;
    int nIO = 4;

    /*Main loop*/
    std::cout<<"Starting PT loop, nbe="<<nbe<<", nbv="<<nbv<<std::endl; 
    int iter;
    double iterror;
    for(iter=1;iter<=niter;iter++){
        
        if (iter==11) tic();
        // PT1<<<gride, blocke>>>(d_Tmp, d_alpha, nbe);
        PT1<<<gride, blocke>>>(d_Tmp, d_alpha_0, d_alpha_1, d_alpha_2, nbe);
        cudaDeviceSynchronize();
    }

    time_s = toc(); double gbs = mem/time_s;

    std::cout<<"Perf: "<<time_s<<" sec. (@ "<<gbs*(iter-10)*nIO<<" GB/s)"<<std::endl;

    /*Copy results from Device to host*/
    cudaMemcpy(vx, d_vx, nbv*sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy(vy, d_vy, nbv*sizeof(double), cudaMemcpyDeviceToHost );

    /*Write output*/
    fid = fopen(outputfile,"wb");
    if (fid==NULL) std::cerr<<"could not open file " << outputfile << " for binary reading or writing";
    WriteData(fid, "PTsolution", "SolutionType");
    WriteData(fid, vx, nbv, 1, "Vx");
    WriteData(fid, vy, nbv, 1, "Vy");
    if (fclose(fid)!=0) std::cerr<<"could not close file " << outputfile;

    /*Cleanup and return*/
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
    // delete [] rheology_B;
    delete [] rheology_B_temp;
    delete [] vx;
    delete [] vy;
    delete [] friction;
    // delete [] alpha2;
    delete [] etan;
    delete [] dVxdt;
    delete [] dVydt;
    delete [] alpha;
    delete [] beta;
    delete [] areas;
    delete [] weights;
    // delete [] resolx;
    // delete [] resoly;
    // delete [] dsdx;
    // delete [] dsdy;
    // delete [] Helem;
    // delete [] ML;
    // delete [] Fvx;
    // delete [] Fvy;
    delete [] Tmp;

    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_alpha);
    cudaFree(d_alpha_0);
    cudaFree(d_alpha_1);
    cudaFree(d_alpha_2);
    cudaFree(dvxdx);
    cudaFree(dvxdy);
    cudaFree(dvydx);
    cudaFree(dvydy);
    cudaFree(d_Tmp);
 

    //Destroying CUDA streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    clean_cuda();
    return 0;
}

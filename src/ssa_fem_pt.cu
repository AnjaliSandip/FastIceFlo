#include <cstdio>
#include <iostream>
#include <string.h>
#include <cmath>
using namespace std;

/*define GPU specific variables*/
#define GPU_ID    0

#define BLOCK_Xe 1024  //optimal block size for JKS2e4 and PIG3e4
#define BLOCK_Xv 1024 

// Device norm subroutine
#define blockId       (blockIdx.x)
#define threadId      (threadIdx.x)
#define isBlockMaster (threadIdx.x==0)

#include "helpers.h"

#define div10 0.1
#define div30 0.0333333333
#define div60 0.0166666667

/*CUDA Code*/
__global__ void PT1(int* reorderd, ftype* vx, ftype* vy, ftype* alpha, ftype* beta, int* index,  ftype* kvx, ftype* kvy, ftype* etan,  ftype* Helem, ftype* areas, bool* isice, ftype* Eta_nbe, ftype* rheology_B, ftype n_glen, ftype eta_0, ftype rele,int nbe){ 
    // int ix = blockIdx.x * blockDim.x + threadIdx.x;
    for(int ixt = blockIdx.x * blockDim.x + threadIdx.x; ixt<nbe; ixt += blockDim.x * gridDim.x){ 
        int ix = reorderd[ixt];

        if (isice[ix]){

            ftype Localalpha[3];
            ftype Localbeta[3];
            ftype Localvx[3];
            ftype Localvy[3];
        
            for(int i=0; i<3; i++){
            Localalpha[i] =  alpha[ix*3+i];
            Localbeta[i] =   beta[ix*3+i];
            Localvx[i] =  vx[index[ix*3+i]-1];
            Localvy[i] =  vy[index[ix*3+i]-1];
            }
        
            ftype dvxdx =  Localvx[0]*Localalpha[0] + Localvx[1]*Localalpha[1] + Localvx[2]*Localalpha[2];
            ftype dvxdy =  Localvx[0]* Localbeta[0] + Localvx[1]* Localbeta[1] + Localvx[2]* Localbeta[2];
            ftype dvydx =  Localvy[0]*Localalpha[0] + Localvy[1]*Localalpha[1] + Localvy[2]*Localalpha[2];
            ftype dvydy =  Localvy[0]* Localbeta[0] + Localvy[1]* Localbeta[1] + Localvy[2]* Localbeta[2];

            ftype  eps_xx = dvxdx;
            ftype  eps_yy = dvydy;
            ftype  eps_xy = .5*(dvxdy+dvydx);
            ftype  EII2   = eps_xx*eps_xx + eps_yy*eps_yy + eps_xy*eps_xy + eps_xx*eps_yy;
            ftype  eta_it = 5e+13;

         if (EII2>0.) eta_it = rheology_B[ix]/(2*__powf(EII2,(n_glen-1.)/(2*n_glen)));

        
                etan[ix] = min(__expf(rele*__logf(eta_it) + (1.0-rele)*__logf(etan[ix])),eta_0*1e5);

                //Viscous Deformation//
                ftype tmp_2hele_etan_areas = 2 * Helem[ix] * etan[ix] *areas[ix];
                for (int i = 0; i < 3; i++){
                        kvx[ix*3+i] = tmp_2hele_etan_areas * ((2 * eps_xx + eps_yy) * Localalpha[i]   +  eps_xy *  Localbeta[i] );
                        kvy[ix*3+i] = tmp_2hele_etan_areas * ((2 * eps_yy + eps_xx) * Localbeta[i]    +  eps_xy * Localalpha[i] );            
            }//isice loop 
        }

   
        Eta_nbe[ix] = etan[ix]*areas[ix];
    }

}


__global__ void PT2_x(ftype* kvx, ftype* groundedratio, ftype* areas, int* index, ftype* alpha2, ftype* vx, ftype* gr_a_alpha2, bool* isice,  int nbe){

    for(int ix = blockIdx.x * blockDim.x + threadIdx.x; ix < nbe; ix += blockDim.x * gridDim.x){
        /*Add basal friction*/
        if (groundedratio[ix] > 0.){
             //   if(groundedratio[ix] > 0. && isice[ix]){
            int n3 = ix * 3;

       ftype myLocalIndex[3][3];
            for (int i = 0; i < 3; i++){
              	for (int j = 0; j < 3; j++){
                      int j_index = index[n3 + j] - 1;
                      myLocalIndex[i][j] = gr_a_alpha2[n3 + i] * vx[j_index];
                }
            }


            ftype tempOutput[3];
            for (int k = 0; k < 3; k++){
                tempOutput[k] = kvx[n3 + k];
            }   


            ftype division;
            for (int k = 0; k < 3; k++){
                for (int i = 0; i < 3; i++){
                    for (int j = 0; j < 3; j++){
                      	ftype temp = myLocalIndex[i][j];

                        if (i == j && j == k){
			    division = div10;
                        } else if ((i!=j) && (j!=k) && (k!=i)){
			    division = div60;
                        } else{
			    division = div30;
                        }
			tempOutput[k] = isice[ix] * tempOutput[k] + temp*division;
                    }
                }
            } 
        
             for (int k = 0; k < 3; k++){
                kvx[n3 + k] = tempOutput[k];
            }  
        }//groundedratio loop
    }
}

__global__ void PT2_y(ftype* kvy, ftype* groundedratio, ftype* areas, int* index, ftype* alpha2, ftype* vy, ftype* gr_a_alpha2, bool* isice,  int nbe){

   for(int ix = blockIdx.x * blockDim.x + threadIdx.x; ix < nbe; ix += blockDim.x * gridDim.x){
        /*Add basal friction*/
        if (groundedratio[ix] > 0.){
            int n3 = ix * 3;

            ftype myLocalIndex[3][3];
            for (int i = 0; i < 3; i++){
                for (int j = 0; j < 3; j++){
                      int j_index = index[n3 + j] - 1;
                      myLocalIndex[i][j] = gr_a_alpha2[n3 + i] * vy[j_index];
                }
            }

          ftype tempOutput[3];
            for (int k = 0; k < 3; k++){
                tempOutput[k] = kvy[n3 + k];
            }

            ftype division;
            for (int k = 0; k < 3; k++){
                for (int i = 0; i < 3; i++){
                    for (int j = 0; j < 3; j++){
                      	ftype temp = myLocalIndex[i][j];

                        if (i == j && j == k){
			    division = div10;
                        } else if ((i!=j) && (j!=k) && (k!=i)){
			    division = div60;
                        } else{
			    division = div30;
                        }
			tempOutput[k] = isice[ix] * tempOutput[k] + temp*division;
                    }
                }
            } 

              for (int k = 0; k < 3; k++){
                kvy[n3 + k] = tempOutput[k];
            }  
  

        }//groundedratio loop
    }
}

//Moving to the next kernel: cannot update kvx and perform indirect access, lines 474 and 475, in the same kernel//
__global__ void PT3(ftype* kvx, ftype* kvy, ftype* Eta_nbe,  ftype* eta_nbv,  int* connectivity, int* columns, ftype* weights, ftype* ML, ftype* KVx, ftype* KVy, ftype* Fvx, ftype* Fvy, ftype* dVxdt, ftype* dVydt, ftype* resolx, ftype* resoly, ftype* H, ftype* vx, ftype* vy, ftype* spcvx, ftype* spcvy, ftype* rho_ML, ftype rho, ftype damp, ftype relaxation, ftype eta_b, int nbv){ 
 
    ftype ResVx;
    ftype ResVy;
    ftype dtVx;
    ftype dtVy;
	
    for(int ix = blockIdx.x * blockDim.x + threadIdx.x; ix<nbv; ix += blockDim.x * gridDim.x){

        KVx[ix] = 0.;
        KVy[ix] = 0.;

        int localColumns[8];
        int localConnectivity[8];
        for(int j=0; j<8; j++){
	        localConnectivity[j] = connectivity[(ix * 8  + j)];
	        localColumns[j] = columns[(ix * 8  + j)];
        } 
      
	    
           ftype tmp_KVx = KVx[ix];
           ftype tmp_KVy = KVy[ix];
           ftype tmp_eta_nbv = eta_nbv[ix];

          for(int j=0;j<8;j++){
        
            if (localConnectivity[j] != 0){
                tmp_KVx = tmp_KVx + kvx[((localConnectivity[j])-1) *3 + ((localColumns[(j)]))];
                tmp_KVy = tmp_KVy + kvy[((localConnectivity[j])-1) *3 + ((localColumns[(j)]))];
            }
              
            if (localConnectivity[j] != 0){
                tmp_eta_nbv = tmp_eta_nbv + Eta_nbe[localConnectivity[j]-1];
            }
            else{break;}
        }    

        KVx[ix] = tmp_KVx;
        KVy[ix] = tmp_KVy;

        eta_nbv[ix] =tmp_eta_nbv/weights[ix];   

        /*1. Get time derivative based on residual (dV/dt)*/
	ResVx =  (-KVx[ix] + Fvx[ix])/rho_ML[ix];
	ResVy =  (-KVy[ix] + Fvy[ix])/rho_ML[ix];
        dVxdt[ix] = dVxdt[ix]*damp + ResVx;
        dVydt[ix] = dVydt[ix]*damp + ResVy;

    
        /*2. Explicit CFL time step for viscous flow, x and y directions*/
        ftype tmp_4eta = 4*eta_nbv[ix]*(1.+eta_b)*4.1;

        dtVx = rho*resolx[ix]*resolx[ix]/tmp_4eta;  
        dtVy = rho*resoly[ix]*resoly[ix]/tmp_4eta; 

        /*3. velocity update, vx(new) = vx(old) + change in vx, Similarly for vy*/
        vx[ix] = vx[ix] + relaxation*dVxdt[ix]*dtVx;
        vy[ix] = vy[ix] + relaxation*dVydt[ix]*dtVy;

        /*Apply Dirichlet boundary condition*/
        if (!isnan(spcvx[ix])){
            vx[ix]    = spcvx[ix];
            dVxdt[ix] = 0.;
        }
        if (!isnan(spcvy[ix])){
            vy[ix]    = spcvy[ix];
            dVydt[ix] = 0.;
        }
    }
}



/*Main*/
int main(){

      /*Open input binary file*/
    const char* inputfile  = "./JKS8e4.bin";
    const char* outputfile = "./output.outbin";
    FILE* fid = fopen(inputfile,"rb");
    if(fid==NULL) std::cerr<<"could not open file " << inputfile << " for binary reading or writing";


    /*Get All we need from binary file*/
    int    nbe,nbv,M,N;
    ftype g,rho,rho_w,yts;
    int    *index           = NULL;
    ftype *spcvx           = NULL;
    ftype *spcvy           = NULL;
    ftype *x               = NULL;
    ftype *y               = NULL;
    ftype *H               = NULL;
    ftype *surface         = NULL;
    ftype *base            = NULL;
    ftype *ice_levelset    = NULL;
    ftype *ocean_levelset  = NULL;
    ftype *rheology_B_temp = NULL;
    ftype *vx              = NULL;
    ftype *vy              = NULL;
    ftype *friction        = NULL;
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
    ftype n_glen     = 3.;
    ftype damp       = dmp; //change to 0.992 for JKS1e6 and 0.998 for PIG2e6
    ftype rele       = rela;   
    ftype eta_b      = 0.5;
    ftype eta_0      = 1.e+14/2.;
    int    niter     = 5e6; //5e6
    int    nout_iter  = 100; //100
    ftype epsi       = 3.171e-7;
    ftype relaxation = stability; //change to 0.999 for JKS1e6 and 0.991 for PIG2e6
    //ftype constant = 4*(1.+eta_b)*4.1;
        
    // Ceiling division to get the close to optimal GRID size
    unsigned int GRID_Xe = 1 + ((nbe - 1) / BLOCK_Xe);
    unsigned int GRID_Xv = 1 + ((nbv - 1) / BLOCK_Xv);

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
    ftype* etan = new ftype[nbe];
    for(int i=0;i<nbe;i++) etan[i] = 1.e+14;
    ftype* dVxdt = new ftype[nbv];
    for(int i=0;i<nbv;i++) dVxdt[i] = 0.;
    ftype* dVydt = new ftype[nbv];
    for(int i=0;i<nbv;i++) dVydt[i] = 0.;

    /*Manage derivatives once for all*/
    ftype* alpha   = NULL;
    ftype* beta    = NULL;
    ftype* areas   = NULL;
    ftype* weights = NULL;
    NodalCoeffs(&areas,&alpha,&beta,index,x,y,nbe);
    Weights(&weights,index,areas,nbe,nbv);


    /*MeshSize*/
    ftype* resolx = new ftype[nbv];
    ftype* resoly = new ftype[nbv];
    MeshSize(resolx,resoly,index,x,y,areas,weights,nbe,nbv);

    /*Physical properties once for all*/
    ftype* dsdx = new ftype[nbe];
    ftype* dsdy = new ftype[nbe];
    derive_xy_elem(dsdx,dsdy,surface,index,alpha,beta,nbe);
    ftype* Helem      = new ftype[nbe];
    ftype* rheology_B = new ftype[nbe];
    for(int i=0;i<nbe;i++){
        Helem[i]      = 1./3. * (H[index[i*3+0]-1] + H[index[i*3+1]-1] + H[index[i*3+2]-1]);
        rheology_B[i] = 1./3. * (rheology_B_temp[index[i*3+0]-1] + rheology_B_temp[index[i*3+1]-1] + rheology_B_temp[index[i*3+2]-1]);
    }

    //Initial viscosity//
    ftype* dvxdx   = new ftype[nbe];
    ftype* dvxdy   = new ftype[nbe];
    ftype* dvydx   = new ftype[nbe];
    ftype* dvydy   = new ftype[nbe];

    derive_xy_elem(dvxdx,dvxdy,vx,index,alpha,beta,nbe);
    derive_xy_elem(dvydx,dvydy,vy,index,alpha,beta,nbe);

    for(int i=0;i<nbe;i++){
        ftype eps_xx = dvxdx[i];
        ftype eps_yy = dvydy[i];
        ftype eps_xy = .5*(dvxdy[i]+dvydx[i]);
        ftype EII2 = pow(eps_xx,2) + pow(eps_yy,2) + pow(eps_xy,2) + eps_xx*eps_yy;
        ftype eta_it = 1.e+14/2.;
        if (EII2>0.) eta_it = rheology_B[i]/(2*pow(EII2,(n_glen-1.)/(2*n_glen)));

        etan[i] = min(eta_it,eta_0*1e5);
        if (isnan(etan[i])){ std::cerr<<"Found NaN in etan[i]"; return 1;}
    }

    /*Linear integration points order 3*/
    ftype wgt3[] = { 0.555555555555556, 0.888888888888889, 0.555555555555556 };
    ftype xg3[]  = {-0.774596669241483, 0.000000000000000, 0.774596669241483 };

    /*Compute RHS amd ML once for all*/
    ftype* ML            = new ftype[nbv];
    ftype* Fvx           = new ftype[nbv];
    ftype* Fvy           = new ftype[nbv];
    ftype* groundedratio = new ftype[nbe];
    bool*   isice         = new bool[nbe];     
    ftype level[3];      

    for(int i=0;i<nbv;i++){
        ML[i]  = 0.;
        Fvx[i] = 0.;
        Fvy[i] = 0.;
    }
    for(int n=0;n<nbe;n++){
        /*Lumped mass matrix*/
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                if (i==j)
                 ML[index[n*3+j]-1] += areas[n]/6.;
                else
                 ML[index[n*3+j]-1] += areas[n]/12.;
            }
        }
        /*Is there ice at all in the current element?*/
        level[0] = ice_levelset[index[n*3+0]-1];
        level[1] = ice_levelset[index[n*3+1]-1];
        level[2] = ice_levelset[index[n*3+2]-1];
        if (level[0]<0 || level[1]<0 || level[2]<0){
            isice[n] = true;
        }
        else{
            isice[n] = false;
            for(int i=0;i<3;i++){
                vx[index[n*3+i]-1] = 0.;
                vy[index[n*3+i]-1] = 0.;
            }
            continue;
        }
        /*RHS, 'F ' in equation 22 (Driving Stress)*/
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                if (i==j){
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

    int countIced = 0;
    for(int n=0;n<nbe;n++){
        if(isice[n])
            countIced++;
    }
    int* h_icedOrdered = new int[nbe];
    int notIcedCounter = countIced;
    countIced=0;
    for(int n=0;n<nbe;n++){
        if(isice[n]){
            h_icedOrdered[countIced++]=n;
        }else{
            h_icedOrdered[notIcedCounter++]=n;
        }
    }



    /*RHS (Water pressure at the ice front)*/
    //  ftype level[3];
    for(int n=0;n<nbe;n++){
        /*Determine if there is an ice front there*/
        level[0] = ice_levelset[index[n*3+0]-1];
        level[1] = ice_levelset[index[n*3+1]-1];
        level[2] = ice_levelset[index[n*3+2]-1];
        int count = 0;
        for(int i=0;i<3;i++) if (level[i]<0.) count++;
        if (count==1){
            /*Ok this element has an ice front, get indices of the 2 vertices*/
            int seg1[2] = {index[n*3+0]-1,index[n*3+1]-1};
            int seg2[2] = {index[n*3+1]-1,index[n*3+2]-1};
            int seg3[2] = {index[n*3+2]-1,index[n*3+0]-1};
            int pairids[2];
            if (ice_levelset[seg1[0]]>=0 && ice_levelset[seg1[1]]>=0){
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
            ftype len = sqrt(pow(x[pairids[1]]-x[pairids[0]],2) + pow(y[pairids[1]]-y[pairids[0]],2) );
            ftype nx  = +(y[pairids[1]]-y[pairids[0]])/len;
            ftype ny  = -(x[pairids[1]]-x[pairids[0]])/len;
            /*RHS*/
            for(int gg=0;gg<2;gg++){
                ftype phi1 = (1.0 -xg3[gg])/2.;
                ftype phi2 = (1.0 +xg3[gg])/2.;
                ftype bg = base[pairids[0]]*phi1 + base[pairids[1]]*phi2;
                ftype Hg = H[pairids[0]]*phi1 + H[pairids[1]]*phi2;
                bg = min(bg,0.0);
                Fvx[pairids[0]] = Fvx[pairids[0]] +wgt3[gg]/2*1/2*(-rho_w*g* pow(bg,2)+rho*g*pow(Hg,2))*nx*len*phi1;
                Fvx[pairids[1]] = Fvx[pairids[1]] +wgt3[gg]/2*1/2*(-rho_w*g*pow(bg,2)+rho*g*pow(Hg,2))*nx*len*phi2;
                Fvy[pairids[0]] = Fvy[pairids[0]] +wgt3[gg]/2*1/2*(-rho_w*g*pow(bg,2)+rho*g*pow(Hg,2))*ny*len*phi1;
                Fvy[pairids[1]] = Fvy[pairids[1]] +wgt3[gg]/2*1/2*(-rho_w*g*pow(bg,2)+rho*g*pow(Hg,2))*ny*len*phi2;
            } 
        }
        /*One more thing in this element loop: prepare groundedarea needed later for the calculation of basal friction*/
        level[0] = ocean_levelset[index[n*3+0]-1];
        level[1] = ocean_levelset[index[n*3+1]-1];
        level[2] = ocean_levelset[index[n*3+2]-1];
        if (level[0]>=0. && level[1]>=0. && level[2]>=0.){
            /*Completely grounded*/
            groundedratio[n]=1.;
        }
        else if (level[0]<=0. && level[1]<=0. && level[2]<=0.){
            /*Completely floating*/
            groundedratio[n]=0.;
        }
        else{
            /*Partially floating,*/
            ftype s1,s2;
            if (level[0]*level[1]>0){/*Nodes 0 and 1 are similar, so points must be found on segment 0-2 and 1-2*/
                s1=level[2]/(level[2]-level[1]);
                s2=level[2]/(level[2]-level[0]);
            }
            else if (level[1]*level[2]>0){ /*Nodes 1 and 2 are similar, so points must be found on segment 0-1 and 0-2*/
                s1=level[0]/(level[0]-level[1]);
                s2=level[0]/(level[0]-level[2]);
            }
            else if (level[0]*level[2]>0){/*Nodes 0 and 2 are similar, so points must be found on segment 1-0 and 1-2*/
                s1=level[1]/(level[1]-level[0]);
                s2=level[1]/(level[1]-level[2]);
            }
            else{
                std::cerr<<"should not be here...";
            }

            if (level[0]*level[1]*level[2]>0.){
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
    ftype* alpha2 = new ftype[nbv];
    for(int i=0;i<nbv;i++){
        /*Compute effective pressure*/
        ftype p_ice   = g*rho*H[i];
        ftype p_water = -rho_w*g*base[i];
        ftype Neff    = p_ice - p_water;
        if (Neff<0.) Neff=0.;
        /*Compute alpha2*/
        alpha2[i] = pow(friction[i],2)*Neff;
    }

    //prepare head and next vectors for chain algorithm, at this point we have not seen any of the elements, so just set the head to -1 (=stop)
    int* head = new int[nbv];
    int* next  = new int[3*nbe];
    for(int i=0;i<nbv;i++) head[i] = -1;

    //Now construct the chain
    for(int k=0;k<nbe;k++){
        for(int j=0;j<3;j++){
            int i;
            int p = 3*k+j;       //unique linear index of current vertex in index
            i = index[p];
            next[p] = head[i - 1];
            head[i -1] = p + 1;
        }
    }

    //Note: Index array starts at 0, but the node# starts at 1
    //Now we can construct the connectivity matrix
    int MAXCONNECT = 8;
    int* connectivity = new int[nbv*MAXCONNECT];
    int* columns = new int[nbv*MAXCONNECT];

    for(int i=0;i<nbv;i++){

        /*Go over all of the elements connected to node I*/
        int count = 0;
        int p=head[i];

        //for (int p = head[i]; p != -1; p = next[p]){
        while (p!= -1){

            int k = p / 3 + 1;     //”row" in index
            int j = (p % 3) - 1;   //"column" in index

            if (j==-1){
                j=2;
                k= k -1;}

            //sanity check
            if (index[p-1] !=i+1){
                std::cout << "Error occurred"  << std::endl;;
            }

            //enter element in connectivity matrix
            connectivity[i * MAXCONNECT + count] = k;
            columns[i * MAXCONNECT + count] = j;
            count++;
            p = next[p-1];
        }
    }
  

    ftype* device_maxvalx = new ftype[GRID_Xv];
    ftype* device_maxvaly = new ftype[GRID_Xv];
    for(int i=0;i<GRID_Xv;i++) device_maxvalx[i] = 0.;
    for(int i=0;i<GRID_Xv;i++) device_maxvaly[i] = 0.;

    ftype* rho_ML = new ftype[nbv];
    for (int i = 0;i < nbv ;i++){
	    rho_ML[i] = rho*max(60.0,H[i])*ML[i];
    }


    ftype* gr_a_alpha2 = new ftype[nbe*3];
    for(int i = 0;i < nbe; i++){
         if (groundedratio[i] > 0.){
            int n3 = i * 3;
            ftype tmp_gr_a = groundedratio[i] * areas[i];

            for (int j = 0; j < 3; j++){
                 int i_index = index[n3 + j] - 1;
                 gr_a_alpha2[n3 + j] =  tmp_gr_a * alpha2[i_index];
            }
         }
    }

    /*------------ now copy all relevant vectors from host to device ---------------*/
    int *d_index = NULL;
    cudaMalloc(&d_index, nbe*3*sizeof(int));
    cudaMemcpy(d_index, index, nbe*3*sizeof(int), cudaMemcpyHostToDevice);

    ftype *d_rho_ML;
    cudaMalloc(&d_rho_ML, nbv*sizeof(ftype));
    cudaMemcpy(d_rho_ML, rho_ML, nbv*sizeof(ftype), cudaMemcpyHostToDevice);

    ftype *d_gr_a_alpha2;
    cudaMalloc(&d_gr_a_alpha2, 3*nbe*sizeof(ftype));
    cudaMemcpy(d_gr_a_alpha2, gr_a_alpha2, 3*nbe*sizeof(ftype), cudaMemcpyHostToDevice);

    ftype *d_vx;
    cudaMalloc(&d_vx, nbv*sizeof(ftype));
    cudaMemcpy(d_vx, vx, nbv*sizeof(ftype), cudaMemcpyHostToDevice);  

    ftype *d_vy;
    cudaMalloc(&d_vy, nbv*sizeof(ftype));
    cudaMemcpy(d_vy, vy, nbv*sizeof(ftype), cudaMemcpyHostToDevice);  
    
 
    ftype *d_alpha;
    cudaMalloc(&d_alpha, nbe*3*sizeof(ftype));
    cudaMemcpy(d_alpha, alpha, nbe*3*sizeof(ftype), cudaMemcpyHostToDevice);

    ftype *d_beta;
    cudaMalloc(&d_beta, nbe*3*sizeof(ftype));
    cudaMemcpy(d_beta, beta, nbe*3*sizeof(ftype), cudaMemcpyHostToDevice);

    ftype *d_etan;
    cudaMalloc(&d_etan, nbe*sizeof(ftype));
    cudaMemcpy(d_etan, etan, nbe*sizeof(ftype), cudaMemcpyHostToDevice);  

    ftype *d_rheology_B;
    cudaMalloc(&d_rheology_B, nbe*sizeof(ftype));
    cudaMemcpy(d_rheology_B, rheology_B, nbe*sizeof(ftype), cudaMemcpyHostToDevice); 

    ftype *d_Helem;
    cudaMalloc(&d_Helem, nbe*sizeof(ftype));
    cudaMemcpy(d_Helem, Helem, nbe*sizeof(ftype), cudaMemcpyHostToDevice); 

    ftype *d_areas;
    cudaMalloc(&d_areas, nbe*sizeof(ftype));
    cudaMemcpy(d_areas, areas, nbe*sizeof(ftype), cudaMemcpyHostToDevice); 

    ftype *d_weights;
    cudaMalloc(&d_weights, nbv*sizeof(ftype));
    cudaMemcpy(d_weights, weights, nbv*sizeof(ftype), cudaMemcpyHostToDevice);  

    ftype *d_ML;
    cudaMalloc(&d_ML, nbv*sizeof(ftype));
    cudaMemcpy(d_ML, ML, nbv*sizeof(ftype), cudaMemcpyHostToDevice);  

    ftype *d_Fvx;
    cudaMalloc(&d_Fvx, nbv*sizeof(ftype));
    cudaMemcpy(d_Fvx, Fvx, nbv*sizeof(ftype), cudaMemcpyHostToDevice); 

    ftype *d_Fvy;
    cudaMalloc(&d_Fvy, nbv*sizeof(ftype));
    cudaMemcpy(d_Fvy, Fvy, nbv*sizeof(ftype), cudaMemcpyHostToDevice); 

    ftype *d_dVxdt;
    cudaMalloc(&d_dVxdt, nbv*sizeof(ftype));
    cudaMemcpy(d_dVxdt, dVxdt, nbv*sizeof(ftype), cudaMemcpyHostToDevice); 

    ftype *d_dVydt;
    cudaMalloc(&d_dVydt, nbv*sizeof(ftype));
    cudaMemcpy(d_dVydt, dVydt, nbv*sizeof(ftype), cudaMemcpyHostToDevice); 

    ftype *d_resolx;
    cudaMalloc(&d_resolx, nbv*sizeof(ftype));
    cudaMemcpy(d_resolx, resolx, nbv*sizeof(ftype), cudaMemcpyHostToDevice);

    ftype *d_resoly;
    cudaMalloc(&d_resoly, nbv*sizeof(ftype));
    cudaMemcpy(d_resoly, resoly, nbv*sizeof(ftype), cudaMemcpyHostToDevice);

    ftype *d_H;
    cudaMalloc(&d_H, nbv*sizeof(ftype));
    cudaMemcpy(d_H, H, nbv*sizeof(ftype), cudaMemcpyHostToDevice);

    ftype *d_spcvx;
    cudaMalloc(&d_spcvx, nbv*sizeof(ftype));
    cudaMemcpy(d_spcvx, spcvx, nbv*sizeof(ftype), cudaMemcpyHostToDevice);

    ftype *d_spcvy;
    cudaMalloc(&d_spcvy, nbv*sizeof(ftype));
    cudaMemcpy(d_spcvy, spcvy, nbv*sizeof(ftype), cudaMemcpyHostToDevice);

    ftype *d_alpha2;
    cudaMalloc(&d_alpha2, nbv*sizeof(ftype));
    cudaMemcpy(d_alpha2, alpha2, nbv*sizeof(ftype), cudaMemcpyHostToDevice);

    ftype *d_groundedratio;
    cudaMalloc(&d_groundedratio, nbe*sizeof(ftype));
    cudaMemcpy(d_groundedratio, groundedratio, nbe*sizeof(ftype), cudaMemcpyHostToDevice);

    bool *d_isice;
    cudaMalloc(&d_isice, nbe*sizeof(bool));
    cudaMemcpy(d_isice, isice, nbe*sizeof(bool), cudaMemcpyHostToDevice);

    int *d_connectivity = NULL;
    cudaMalloc(&d_connectivity, nbv*8*sizeof(int));
    cudaMemcpy(d_connectivity, connectivity, nbv*8*sizeof(int), cudaMemcpyHostToDevice);

    int *d_columns = NULL;
    cudaMalloc(&d_columns, nbv*8*sizeof(int));
    cudaMemcpy(d_columns, columns, nbv*8*sizeof(int), cudaMemcpyHostToDevice);

    ftype* d_device_maxvalx = NULL;
    cudaMalloc(&d_device_maxvalx, GRID_Xv*sizeof(ftype));
    cudaMemcpy(d_device_maxvalx, device_maxvalx, GRID_Xv*sizeof(ftype), cudaMemcpyHostToDevice);

    ftype* d_device_maxvaly = NULL;
    cudaMalloc(&d_device_maxvaly, GRID_Xv*sizeof(ftype));
    cudaMemcpy(d_device_maxvaly, device_maxvaly, GRID_Xv*sizeof(ftype), cudaMemcpyHostToDevice); 

    int *d_icedOrdered = NULL;
    cudaMalloc(&d_icedOrdered, nbe*sizeof(int));
    cudaMemcpy(d_icedOrdered, h_icedOrdered, nbe*sizeof(int), cudaMemcpyHostToDevice);


    /*------------ allocate relevant vectors on host (GPU)---------------*/
    cudaMalloc(&dvxdx,nbe*sizeof(ftype));

    cudaMalloc(&dvxdy, nbe*sizeof(ftype));

    cudaMalloc(&dvydx, nbe*sizeof(ftype));

    cudaMalloc(&dvydy, nbe*sizeof(ftype));

    ftype *KVx = NULL;
    cudaMalloc(&KVx, nbv*sizeof(ftype));

    ftype *KVy = NULL;
    cudaMalloc(&KVy, nbv*sizeof(ftype));

    ftype *eta_nbv = NULL;
    cudaMalloc(&eta_nbv, nbv*sizeof(ftype));

    ftype *Eta_nbe = NULL;
    cudaMalloc(&Eta_nbe, nbe*3*sizeof(ftype));

    ftype *kvx = NULL;
    cudaMalloc(&kvx, nbe*3*sizeof(ftype));

    ftype *kvy = NULL;
    cudaMalloc(&kvy, nbe*3*sizeof(ftype));
    
    //Creating CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Perf
    ftype time_s = 0.0;
    ftype mem = (ftype)1e-9*(ftype)nbv*sizeof(ftype);
    int nIO = 8;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    /*Main loop*/
    std::cout<<"Starting PT loop, nbe="<<nbe<<", nbv="<<nbv<<std::endl; 
    int iter;
    ftype iterror;
    for(iter=1;iter<=niter;iter++){
      
       if (iter==11) cudaEventRecord(start);

        PT1<<<gride, blocke, 0, stream1>>>(d_icedOrdered,d_vx, d_vy, d_alpha, d_beta, d_index, kvx,  kvy, d_etan, d_Helem, d_areas, d_isice, Eta_nbe, d_rheology_B, n_glen, eta_0, rele, nbe);
        cudaStreamSynchronize(stream1);

        PT2_x<<<gride, blocke, 0, stream1>>>(kvx, d_groundedratio, d_areas, d_index, d_alpha2, d_vx, d_gr_a_alpha2, d_isice, nbe);
        cudaStreamSynchronize(stream1);
        PT2_y<<<gride, blocke, 0, stream2>>>(kvy, d_groundedratio, d_areas, d_index, d_alpha2, d_vy, d_gr_a_alpha2, d_isice, nbe);
        cudaStreamSynchronize(stream2);
       

        PT3<<<gridv, blockv, 0, stream1>>>(kvx, kvy, Eta_nbe, eta_nbv, d_connectivity, d_columns, d_weights, d_ML, KVx, KVy, d_Fvx, d_Fvy, d_dVxdt, d_dVydt, d_resolx, d_resoly, d_H, d_vx, d_vy, d_spcvx, d_spcvy, d_rho_ML, rho, damp, relaxation, eta_b, nbv);
        cudaStreamSynchronize(stream1);

       if ((iter % nout_iter) == 0){
            //Get final error estimate/
            __device_max_x(dVxdt); 
            __device_max_y(dVydt);
            if (isnan(device_MAXx) || isnan(device_MAXy)) {break;}
            else {
            iterror = max(device_MAXx, device_MAXy);
            std::cout<<"iter="<<iter<<", err="<<iterror<<std::endl;
            if ((iterror < epsi) && (iter > 100)) break;
       } 
      }  
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float seconds = 0;
    cudaEventElapsedTime(&seconds, start, stop);
    seconds = seconds/1000.0;

    time_s = seconds; ftype gbs = mem/time_s;
    std::cout<<"\n Perf: "<<time_s<<" sec. (@ "<<gbs*(iter-10)*nIO<<" GB/s)"<<std::endl;

    /*Copy results from Device to host*/
    cudaMemcpy(vx, d_vx, nbv*sizeof(ftype), cudaMemcpyDeviceToHost );
    cudaMemcpy(vy, d_vy, nbv*sizeof(ftype), cudaMemcpyDeviceToHost );
 
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
    delete [] rho_ML;
    delete [] gr_a_alpha2;
    delete [] h_icedOrdered;

    cudaFree(d_index);
    cudaFree(d_rho_ML);
    cudaFree(d_gr_a_alpha2);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_etan);
    cudaFree(d_rheology_B);
    cudaFree(d_Helem);
    cudaFree(d_areas);
    cudaFree(d_weights);
    cudaFree(d_ML);
    cudaFree(d_Fvx);
    cudaFree(d_Fvy);
    cudaFree(d_dVxdt);
    cudaFree(d_dVydt);
    cudaFree(d_resolx);
    cudaFree(d_resoly);
    cudaFree(d_H);
    cudaFree(d_spcvx);
    cudaFree(d_spcvy);   
    cudaFree(d_alpha2);
    cudaFree(d_groundedratio);
    cudaFree(d_isice);
    cudaFree(d_connectivity);
    cudaFree(d_columns);
    cudaFree(dvxdx);
    cudaFree(dvxdy);
    cudaFree(dvydx);
    cudaFree(dvydy);
    cudaFree(KVx);
    cudaFree(KVy);
    cudaFree(eta_nbv);
    cudaFree(Eta_nbe);
    cudaFree(kvx);
    cudaFree(kvy);
    cudaFree(d_device_maxvalx);
    cudaFree(d_device_maxvaly);
    cudaFree(d_icedOrdered);

    //Destroying CUDA streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    //Destroying CUDA destroy
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    clean_cuda();
    return 0;
}

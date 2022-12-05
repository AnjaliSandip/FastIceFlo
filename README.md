# PT FEM SSA GPU solver


Copyright (C) 2022 Anjali Sandip, Ludovic Raess and and Mathieu Morlighem.

XXX is a free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

XXX is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with XXX. If not, see http://www.gnu.org/licenses/.

Current version can be found:  here??


- Aim of the project
- Citation information
- Conservation of Momentum Balance equations (+ PT Solver + FEM method
- contact: anjali.sandip@und.edu
-

# Ice-sheet flow/2-D Shallow shelf approximation (SSA)
We employ SSA \citep{macayeal1989large}) to solve the momentum balance to predict ice-sheet flow. The SSA equations in the matrix form read : <br>
$\nabla \cdot \left(2 H \mu \dot{\boldsymbol{\varepsilon}}_{SSA} \right) = \rho g H\nabla s  + \alpha^2 {\bf v}$


# Pseudo-transient method

# Ice-sheet flow model configurations 

![gmd_domain](https://user-images.githubusercontent.com/60862184/204933517-d4b81b5b-acb3-4256-a8be-02439db7f3dc.png)

Figure 1. Glacier model configurations; observed surface velocities interpolated on a uniform mesh. Panels $\textbf{(a)}$ and $\textbf{(b)}$  correspond to Jakobshavn Isbrae and Pine Island Glacier respectively.

Step 1. Install [ISSM](https://issm.jpl.nasa.gov/download/) <br>
Step 2. Run "runme.m" script (located in PIG and JKS folders) <br>
 
| DoFs |  Jakobshavn Isbrae resol (m) | DoFs | Pine Island Glacier resol (m)|       
| :----: | :----: | :----: | :----: | 
| 8e4 | 600 | 3e4 | 2500 | 
| 3e5 | 310 | 7e4 | 1750 |
| 7e5 | 200 | 1e5 | 1250 |
| 1e6 | 170 | 2e6 |  300 |
| 2e7 | ?? |

Table 1.  Average element size or spatial resolution "resol" for the glacier model configurations chosen in the study <br>

Step 3. Save the .mat file and corresponding .bin file
`md=solve(md,'Stressbalance')`

# High-spatial-resolution ice-sheet flow simulations on GPUs
To perform the numerical experiments described in the pre-print,  <br>
Step 1. Clone or download this repository.  <br>
Step 2. Compile the `ssa_fem_pt.cu` routine on a system hosting an Nvidia Tesla V100 GPU `nvcc -arch=sm_70 -O3 -lineinfo   ssa_fem_pt.cu  -Ddmp=$damp -Dstability=$vel_rela -Drela=$visc_rela`   <br>
Step 3. Run  <br>
Step 4. Extract and plot the results, ice velocity distribution, for a glacier model configuration at a spatial resolution (or grid size): <br>
        -  Store .mat file and corresponding outbin file in a MATLAB directory <br>
        -  Run the visme.m file from the same directory (as in Step 1) <br>
        -  View results <br>
        

| Jakobshavn Isbrae DoFs | $\gamma$  | $\theta_v$ | $\theta_{\mu}$ | Pine Island Glacier DoFs | $\gamma$ | $\theta_v$ | $\theta_{\mu}$ |
| :----: | :----: | :----: | :----: |:----: | :----: | :----: | :----: |
| 8e4 | 0.98 | 0.99 | 3e-2 | 3e4 | 0.981 | 0.967 | 7e-2 |
| 3e5 | 0.99 | 0.9 | 3e-2 | 7e4 | 0.99 | 0.9 | 3e-2 |
| 7e5 | 0.99 | 0.99 | 1e-1 | 1e5 | 0.99 | 0.993 | 1e-1 |
| 1e6 | 0.992 | 0.999 | 1e-1 | 2e6 | 0.998 | 0.991 | 1e-1 |
| 2e7 | 0.992 | 0.999 | 1e-1 |

Table 2. Optimal combination of damping parameter $\gamma$,  non-linear viscosity relaxation scalar $\theta_{\mu}$ and relaxation $\theta_v$  to maintain the linear scaling and solution stability for the glacier model configurations and DoFs listed below.


We compared the CUDA C PT implementation with the Krylov subspace method relying on biconjugate gradient with block Jacobi pre-conditioner (bcgsl/bjacobi). The chosen CPU architecture was a 64-bit 18-core Intel Xeon Gold 6140 processor with 192 GB of RAM per node. CPU-based multi-core MPI-parallelized ice-sheet flow simulations were executed on two CPU processors, all 36 cores enabled.

In order to assess the performance of the memory-bound PT algorithm on Ampere A100 SXM4 featuring 80GB on-board memory, we employ the effective memory throughput metric  ${\bf T}_{eff}$.  The results are listed below:

| DoFs |  Jakobshavn Isbrae (GB/s)  | DoFs | Pine Island Glacier (GB/s)|       
| :----: | :----: | :----: | :----: | 
| 8e4 | 34 | 3e4 | 17 | 
| 3e5 | 47 | 7e4 | 30 |
| 7e5 | 58 | 1e5 | 36 |
| 1e6 | 56 | 2e6 | 58 |
| 2e7 | 38 | 3e7 | 36 |


# FastIceFlo

[![DOI](https://zenodo.org/badge/xxx.svg)](https://zenodo.org/badge/latestdoi/xxx)

This study aims to provide a GPU-friendly matrix-free and iterative algorithm to solve momentum balance for unstructured meshes to predict ice-sheet flow for actual glaciers. 

The insights from this study will benefit efforts to diminish spatial resolution constraints and increase computing speed for unstructured mesh applications, including but not limited to ice-sheet modeling.

This repository relates to the original research article published in the **Geoscientific Model Development** journal:
```
@Article{gmd-xx,
    AUTHOR = {Sandip, A. and R\"ass, L. and Morlighem, M.},
    TITLE = {Graphics processing unit accelerated ice flow solver for unstructured meshes using the Shallow Shelf Approximation (FastIceFlo v1.0)},
    JOURNAL = {Geoscientific Model Development},
    VOLUME = {xx},
    YEAR = {xx},
    NUMBER = {xx},
    PAGES = {xx--xx},
    URL = {xx},
    DOI = {xx}
}
```


## 2-D Shallow shelf approximation (SSA)
We employ SSA to solve the momentum balance to predict ice-sheet flow:

$\nabla \cdot \left(2 H \mu \dot{\boldsymbol{\varepsilon}} \right) = \rho g H\nabla s  + \alpha^2 {\bf v}$ ,

where $H$ is the ice thickness distribution, $\mu$ the dynamic ice viscosity, $\dot{\boldsymbol{\varepsilon}}$ the effective strain rate, $\rho$ the ice density, $g$ the gravitational acceleration, $s$ glacier's upper surface z-coordinate and $\alpha^2 {\bf v}$ is the basal friction.

As boundary conditions, we apply water pressure at the ice front $\Gamma_{\sigma}$, and non-homogeneous Dirichlet boundary conditions on the other boundaries $\Gamma_u$ (based on observed velocity).

## Pseudo-transient (PT) method
We reformulate the 2-D SSA steady-state momentum balance equations to incorporate the usually ignored inertial terms:

$\nabla \cdot \left(2 H \mu \dot{\boldsymbol{\varepsilon}}_{SSA} \right) -\rho g H\nabla s  - \alpha^2 {\bf v} = \rho H\frac{\partial \bf v}{\partial \tau}$ ,

which allows us to turn the steady-state equations into transient diffusion of velocities $v_{x,y}$. The velocity-time derivatives represent physically motivated expressions we can further use to iteratively reach a steady state, thus the solution of the system.

## Weak form
The weak form (assuming homogeneous Dirichlet conditions along all model boundaries for simplicity) reads:

$\forall {\bf w}\in {\mathcal H}^1\left(\Omega\right)$ ,

$\int_\Omega {\rho} H\frac{\partial {\bf v}}{\partial \tau} \cdot {\bf w} d\Omega$ + 
$\int_\Omega 2 H {\mu} \dot{\boldsymbol{\varepsilon}} : \dot{\boldsymbol{\varepsilon}}_{w} d\Omega$ =

$\int_\Omega  - \rho g H \nabla s \cdot {\bf w} - \alpha^2 {\bf v} \cdot {\bf w} d\Omega$

where ${\mathcal H}^1\left(\Omega\right)$ is the space of square-integrable functions whose first derivatives are also square integrable. 

Once discretized using the finite-element method, the matrix system to solve is:

$\boldsymbol{M} \dot{\bf V} + \boldsymbol{K}{\bf V} = \boldsymbol{K}$ ,

where $\boldsymbol{M}$ is the mass matrix, $\boldsymbol{K}$ is the stiffness matrix, $\boldsymbol{F}$ is the right hand side or load vector, and ${\bf V}$ is the vector of ice velocity.

For every nonlinear PT iteration, we compute the rate of change in velocity $\dot{\bf v}$ and the explicit CFL time step $\Delta \tau$. We then deploy the reformulated 2D SSA momentum balance equations  to update ice velocity $\bf v$ followed by ice viscosity $\mu_{eff}$.  [We iterate in pseudo-time until the stopping criterion is met](docs/fig_pt_flowchart.pdf).


## Step 1: Generate glacier model configurations 
To test the performance of the PT method beyond simple idealized geometries, we apply it to two regional-scale glaciers: [Jakobshavn Isbræ, in western Greenland, and Pine IslandGlacier, in west Antarctica](docs/fig_gmd.pdf).

To generate the glacier model configurations, follow the steps listed below:
1. Install [ISSM](https://issm.jpl.nasa.gov/download/)
2. Run `runme.m` script to generate the [Jakobshavn Isbrae](BinFileGeneration/JKS/runme.m) or [Pine Island](BinFileGeneration/PIG/runme.m) Glacier models
3. Save the .mat file and corresponding .bin file

## Step 2: Hardware implementation
We developed a CUDA C implementation to solve the SSA equations using the PT approach on unstructured meshes. To execute on a NVIDIA Tesla V100 GPU and view results, follow the steps listed below:

1. Clone or download this repository.
2. Transfer the .bin file generated in the previous step along with files in [src](src) folder to a directory 
3. Compile the [`ssa_fem_pt.cu`](src/ssa_fem_pt.cu) routine on a system hosting a (recent) Nvidia CUDA-capable GPU (here shown for a Tesla V100)
```bash
nvcc -arch=sm_70 -O3 -lineinfo   ssa_fem_pt.cu  -Ddmp=$damp -Dstability=$vel_rela -Drela=$visc_rela
```
3. Run the generated executable `./a.out`
4. Along with a `.txt` file that stores the computational time, effective memory throughput and the PT iterations to meet stopping criterion, a `.outbin` file will be generated.

## Step 3: Post-processing
To extract and plot the ice velocity distribution, follow the steps listed below:
 1. Store `.mat` file (Glacier model configurations "step 3") and the `.outbin` file (Hardware implementation "step 4") in a MATLAB directory
 2. In the ISSM environment, execute the following statements in the MATLAB command window:
        ```Matlab
        load "insert name of .mat file here"
        md.miscellaneous.name = 'output';
        md=loadresultsfromdisk(md, 'output.outbin')
        plotmodel(md,'data',sqrt(md.results.PTsolution.Vx.^2 + md.results.PTsolution.Vy.^2));
        ```
  3. View results

| Jakobshavn Isbrae number of vertices | $\gamma$  | $\theta_v$ | $\theta_{\mu}$ | Block size |
| :----: | :----: | :----: | :----: |:----: |
| 44229 | 0.98 | 0.99 | 3e-2 | 128 | 
| 167337 | 0.99 | 0.9 | 3e-2 | 128 | 
| 393771 | 0.99 | 0.99 | 1e-1 | 1024 |
| 667729 | 0.992 | 0.999 | 1e-1 | 1024 |
| 10664257 | 0.998 | 0.999 | 1e-1 | 1024 |

| Pine Island Glacier number of vertices | $\gamma$ | $\theta_v$ | $\theta_{\mu}$ |Block size |
| :----: | :----: | :----: | :----: |:----: |
| 17571 | 0.981 | 0.967 | 7e-2 | 128 |
| 35646 | 0.99 | 0.9 | 3e-2 | 256 |
| 69789 | 0.991 | 0.99 | 2e-2 | 512 |
| 1110705 | 0.998 | 0.995 | 1e-2 | 1024 |


Table 2. Optimal combination of damping parameter $\gamma$,  non-linear viscosity relaxation scalar $\theta_{\mu}$ and relaxation $\theta_v$  to maintain the linear scaling and solution stability for the glacier model configurations and DoFs listed below. Optimal block size was chosen to minimize wall time.

## Performance metric
In order to assess the performance of the memory-bound PT CUDA C implementation on NVIDIA Tesla V100 SXM2 GPU (16GB DRAM) and a Tesla A100 SXM4 (40GB DRAM) we employ the **effective memory throughput** metric.


|Jakobshavn Isbrae number of vertices | V100 (GB/s) | A100 (GB/s)|
| :----: | :----: | :----: | 
| 44229 | 24 | 34 | 
| 167337 | 23 | 47 |
| 393771 | 23 | 58 |
| 667729 | 19 | 56 |
| 10664257 | 11 | 38 |

|Pine Island Glacier number of vertices | V100 (GB/s) | A100 (GB/s)|
| :----: | :----: | :----: |  
| 17571 | 17 | 17 | 
| 35646 | 23 | 30 |
| 69789 | 23 | 36 |
| 1110705 | 18 | 58 |
| 17747649 | 11 | 36 |

We compared the effective memory throughput with the peak, i.e. the memory transfer speed for performing memory copy operations only. It represents the hardware performance limit. The code [`memcopy.cu`](scripts/memcopy.cu) located in the [scripts](scripts) directory reports the peak memory throughput. The reported peak memory throughput for the GPU hardware NVIDIA Tesla V100 and NVIDIA A100 were 785 GB/s and 1536 GB/s respectively.  

## Questions/Comments/Discussion
For questions, comments and discussions please post in the FastIceFlo discussions forum.

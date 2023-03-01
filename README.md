# FastIceFlo

[![DOI](https://zenodo.org/badge/xxx.svg)](https://zenodo.org/badge/latestdoi/xxx)

This study aims to provide a graphics processing unit accelerated ice flow solver for unstructured meshes using the Shallow Shelf Approximation.


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

### Pseudo-transient (PT) method
We reformulate the 2-D SSA steady-state momentum balance equations to incorporate the usually ignored inertial terms:

$\nabla \cdot \left(2 H \mu \dot{\boldsymbol{\varepsilon}}_{SSA} \right) -\rho g H\nabla s  - \alpha^2 {\bf v} = \rho H\frac{\partial \bf v}{\partial \tau}$ ,

which allows us to turn the steady-state equations into transient diffusion of velocities $v_{x,y}$. The velocity-time derivatives represent physically motivated expressions we can further use to iteratively reach a steady state, thus the solution of the system.

### Weak form
The weak form (assuming homogeneous Dirichlet conditions along all model boundaries for simplicity) reads:

$\forall {\bf w}\in {\mathcal H}^1\left(\Omega\right)$ ,

$\int_\Omega {\rho} H\frac{\partial {\bf v}}{\partial \tau} \cdot {\bf w} d\Omega$ + 
$\int_\Omega 2 H {\mu} \dot{\boldsymbol{\varepsilon}} : \dot{\boldsymbol{\varepsilon}}_{w} d\Omega$ =


$\int_\Omega  - \rho g H \nabla s \cdot {\bf w} - \alpha^2 {\bf v} \cdot {\bf w} d\Omega$

where ${\mathcal H}^1\left(\Omega\right)$ is the space of square-integrable functions whose first derivatives are also square integrable. 

Once discretized using the finite-element method, the matrix system to solve is:

$\boldsymbol{M} \dot{\bf V} + \boldsymbol{K}{\bf V} = \boldsymbol{F}$ ,

where $\boldsymbol{M}$ is the mass matrix, $\boldsymbol{K}$ is the stiffness matrix, $\boldsymbol{F}$ is the right hand side or load vector, and ${\bf V}$ is the vector of ice velocity.

For every nonlinear PT iteration, we compute the rate of change in velocity $\dot{\bf v}$ and the explicit CFL time step $\Delta \tau$. We then deploy the reformulated 2D SSA momentum balance equations  to update ice velocity $\bf v$ followed by ice viscosity $\mu_{eff}$.  [We iterate in pseudo-time until the stopping criterion is met](docs/fig_pt_flowchart.pdf).

## Steps to run the code
### Step 1: Generate glacier model configurations 
To test the performance of the PT method beyond simple idealized geometries, we apply it to two regional-scale glaciers: [Jakobshavn Isbræ, in western Greenland, and Pine IslandGlacier, in west Antarctica](docs/fig_gmd.pdf).

To generate the glacier model configurations, follow the steps listed below:
1. Install [ISSM](https://issm.jpl.nasa.gov/download/) and download the datasets
2. Run `runme.m` script to generate the [Jakobshavn Isbræ](BinFileGeneration/JKS/runme.m) or [Pine Island](BinFileGeneration/PIG/runme.m) Glacier models
3. Save the `.mat` file and corresponding `.bin` file

### Step 2: Hardware implementation
We developed a CUDA C implementation to solve the SSA equations using the PT approach on unstructured meshes.  We choose a stopping criterion of $||v^{old} - v||_{\infty}$ < 10 m $yr^{-1}$. To execute on a NVIDIA Tesla V100 GPU and view results, follow the steps listed below:

1. Clone or download this repository.
2. Transfer the `.bin` file generated along with files in [src](src) folder to a directory on a system hosting a (recent) Nvidia CUDA-capable GPU (here shown for a Tesla V100)
3. Plug in the damping parameter $\gamma$,  non-linear viscosity relaxation scalar $\theta_{\mu}$ and relaxation $\theta_v$ for the chosen glacier model configuration and spatial resolution
4. Compile the [`ssa_fem_pt.cu`](src/ssa_fem_pt.cu) routine 
```bash
nvcc -arch=sm_70 -O3 -lineinfo   ssa_fem_pt.cu  -Ddmp=$damp -Dstability=$vel_rela -Drela=$visc_rela
```
5. Run the generated executable `./a.out`
6. Along with a `.txt` file that stores the computational time, effective memory throughput and the PT iterations to meet stopping criterion, a `.outbin` file will be generated.
7. Save the `.outbin` file

| Jakobshavn Isbræ number of vertices | $\gamma$  | $\theta_v$ | $\theta_{\mu}$ | Block size |
| :----: | :----: | :----: | :----: |:----: |
| 44229 | 0.98 | 0.99 | 3e-2 | 128 | 
| 164681 | 0.987 | 0.98 | 7e-2 | 128 | 
| 393771 | 0.99 | 0.99 | 1e-1 | 1024 |
| 667729 | 0.992 | 0.999 | 1e-1 | 1024 |
| 10664257 | 0.998 | 0.999 | 1e-1 | 1024 |

| Pine Island Glacier number of vertices | $\gamma$ | $\theta_v$ | $\theta_{\mu}$ |Block size |
| :----: | :----: | :----: | :----: |:----: |
| 14460 | 0.98 | 0.6 | 1e-1 | 128 |
| 35646 | 0.99 | 0.49 | 8e-2 | 256 |
| 69789 | 0.991 | 0.99 | 2e-2 | 512 |
| 1110705 | 0.998 | 0.995 | 1e-2 | 1024 |


Table 1. Optimal combination of damping parameter $\gamma$,  non-linear viscosity relaxation scalar $\theta_{\mu}$ and relaxation $\theta_v$  to maintain the linear scaling and solution stability for the glacier model configurations and DoFs listed. Optimal block size was chosen to minimize wall time.

### Step 3: Post-processing
To extract and plot the ice velocity distribution, follow the steps listed below:
 1. Transfer `.mat` file and the `.outbin` files generated from steps 1 and 2 to a directory in MATLAB
 2. Activate the ISSM environment
 3. Execute the following statements in the MATLAB command window:
        ```Matlab
        load "insert name of .mat file here" 
        md.miscellaneous.name = 'output'; 
        md=loadresultsfromdisk(md, 'output.outbin'); 
        plotmodel(md,'data',sqrt(md.results.PTsolution.Vx.^2 + md.results.PTsolution.Vy.^2));
        ```
  3. View results

## Questions/Comments/Discussion
For questions, comments and discussions please post in the FastIceFlo discussions [discussions](https://github.com/AnjaliSandip/FastIceFlo/discussions) forum.

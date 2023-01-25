
This study aims to provide a GPU-friendly algorithm to solve momentum balance for unstructured meshes to predict ice-sheet flow for actual glaciers. The solution algorithm is matrix-free and iterative. The insights from this study will benefit efforts to diminish spatial resolution constraints and increase computing speed for unstructured mesh applications, including but not limited to ice-sheet modeling.


- Citation information


# 2-D Shallow shelf approximation (SSA)
We employ SSA to solve the momentum balance to predict ice-sheet flow. The SSA equations in the matrix form read : <br>
$\nabla \cdot \left(2 H \mu \dot{\boldsymbol{\varepsilon}}_{SSA} \right) = \rho g H\nabla s  + \alpha^2 {\bf v}$

where $H$  is the ice thickness distribution, $\mu$ the dynamic ice viscosity, $\dot{\boldsymbol{\varepsilon}}_{SSA}$ the effective strain rate, $\rho$ the ice density, $g$ the gravitational acceleration, $s$ glacier's upper surface z-coordinate and $\alpha^2 {\bf v}$ is the basal friction. <br>

In terms of  boundary conditions, we apply water pressure  at the ice front $\Gamma_{\sigma}$, and non-homogeneous Dirichlet boundary conditions on the other boundaries $\Gamma_u$ (based on observed velocity). 

# Pseudo-transient (PT) method
We reformulate the 2-D SSA steady-state momentum balance equations to incorporate the usually ignored inertial terms: <br>
$\nabla \cdot \left(2 H \mu \dot{\boldsymbol{\varepsilon}}_{SSA} \right) -\rho g H\nabla s  - \alpha^2 {\bf v} = \rho H\frac{\partial \bf v}{\partial \tau}$

allows us to turn the steady-state equations into transient diffusion of velocities $v_{x,y}$. The velocity-time derivatives represent physically motivated expressions we can further use to iteratively reach a steady state, thus the solution of the system.

# Weak form
The weak form (assuming homogeneous Dirichlet conditions along all model boundaries for simplicity) is: $\forall {\bf w}\in {\mathcal H}^1\left(\Omega\right)$, <br>
$\int_\Omega {\rho} H\frac{\partial {\bf v}}{\partial \tau} \cdot {\bf w}d\Omega$ +  \int_\Omega  2 H \mu \dot{\boldsymbol{\varepsilon}}_{SSA}: \dot{\boldsymbol{\varepsilon}}_{w} \;d\Omega$ <br>
$= \int_\Omega  - \rho g H \nabla s \cdot {\bf w} - \alpha^2 {\bf v} \cdot {\bf w} \;d\Omega$

where ${\mathcal H}^1\left(\Omega\right)$ is the space of square-integrable functions whose first derivatives are also square integrable. 

Once discretized using the finite-element method, the matrix system to solve is:
$\boldsymbol{M} \dot{\bf V} + \boldsymbol{K}{\bf V} = \boldsymbol{K}$ <br>

where $\boldsymbol{M}$ is the mass matrix, $\boldsymbol{K}$ is the stiffness matrix, $\boldsymbol{F}$ is the right hand side or load vector, and ${\bf V}$ is the vector of ice velocity.

For every nonlinear PT iteration,  we compute the rate of change in velocity $\dot{\bf v}$ and the explicit CFL time step $\Delta \tau$. We then deploy the reformulated 2D SSA momentum balance equations  to update ice velocity $\bf v$ followed by ice viscosity $\mu_{eff}$.  We iterate in pseudo-time until the stopping criterion is met. <br>
![Screenshot from 2023-01-23 17-15-51](https://user-images.githubusercontent.com/60862184/214173707-a8d442a9-8933-49ec-8b6b-806212e7a8d2.png)

# Glacier model configurations 

![gmd_domain](https://user-images.githubusercontent.com/60862184/204933517-d4b81b5b-acb3-4256-a8be-02439db7f3dc.png)

Figure 1. Glacier model configurations; observed surface velocities interpolated on a uniform mesh. Panels $\textbf{(a)}$ and $\textbf{(b)}$  correspond to Jakobshavn Isbrae and Pine Island Glacier respectively.

Step 1. Install [ISSM](https://issm.jpl.nasa.gov/download/) <br>
Step 2. Run "runme.m" script (located in BinFileGeneration/PIG or JKS folders). For example, <br>
 
| DoFs |  Jakobshavn Isbrae resol (m) | DoFs | Pine Island Glacier resol (m)|       
| :----: | :----: | :----: | :----: | 
| 8e4 | 600 | 7e4 | 1750 | 

Table 1.  Average element size or spatial resolution "resol" for the glacier model configurations chosen in the study <br>

Step 3. Save the .mat file and corresponding .bin file
![JKS8e4](https://user-images.githubusercontent.com/60862184/214713810-7d731d8f-862e-4199-b1bf-2e0bacbf282d.png)



# Hardware implementation
Step 1. Clone or download this repository.  <br>
Step 2. Compile the `ssa_fem_pt.cu` routine on a system hosting an Nvidia Tesla V100 GPU <br>
`nvcc -arch=sm_70 -O3 -lineinfo   ssa_fem_pt.cu  -Ddmp=$damp -Dstability=$vel_rela -Drela=$visc_rela`   <br>
Step 3. Run `./a.out` <br>
Step 4. Along with a .txt file that stores the computational time, effective memory throughput and the PT iterations to meet stopping criterion, a .outbin file will be generated.  To extract and plot the ice velocity distribution, for a glacier model configuration at a spatial resolution (or grid size): <br>
         4.1 Store .mat file (Glacier model configurations/step 3) and the.outbin file in a MATLAB directory <br>
         4.2 Execute the following statements in the MATLAB command window: <br>
        `load "insert name of .mat file here"`  <br>
        `md.miscellaneous.name = 'output';` <br>
        `md=loadresultsfromdisk(md, 'output.outbin')` <br>
        `plotmodel(md,'data',sqrt(md.results.PTsolution.Vx.^2 + md.results.PTsolution.Vy.^2));` <br>
       
   
        
        Figure 2. Jakobshavn Isbrae ice velocity field, at 600 meter spatial resolution. <br>

        4.3 View results <br>
        
| Jakobshavn Isbrae number of vertices | $\gamma$  | $\theta_v$ | $\theta_{\mu}$ | Block size | Pine Island Glacier number of vertices | $\gamma$ | $\theta_v$ | $\theta_{\mu}$ |Block size |
| :----: | :----: | :----: | :----: |:----: | :----: | :----: | :----: | :----: | :----: |
| 44229 | 0.98 | 0.99 | 3e-2 | 128 | 17571 | 0.981 | 0.967 | 7e-2 | 128 |
| 167337 | 0.99 | 0.9 | 3e-2 | 128 | 35646 | 0.99 | 0.9 | 3e-2 | 256 |
| 393771 | 0.99 | 0.99 | 1e-1 | 1024 | 69789 | 0.99 | 0.993 | 1e-1 | 512 |
| 667729 | 0.992 | 0.999 | 1e-1 | 1024 | 1110705 | 0.998 | 0.991 | 1e-1 | 1024 |
| 10664257 | 0.998 | 0.999 | 1e-1 | 1024 |

Table 2. Optimal combination of damping parameter $\gamma$,  non-linear viscosity relaxation scalar $\theta_{\mu}$ and relaxation $\theta_v$  to maintain the linear scaling and solution stability for the glacier model configurations and DoFs listed below. Optimal block size to increase occupancy and reduce wall time.

We compare the PT Tesla V100 GPU implementation with ISSM’s “standard” CPU implementation using a conjugate gradient iterative solver. As a CPU, we used a 64-bit 18-core Intel Xeon Gold 6140 processor with 192 GB of RAM per node. We executed multi-core MPI-parallelized ice-sheet
flow simulations on two CPUs, all 36 cores enabled. We performed computations using double-precision arithmetic. 

The results from GPU and CPU implementations are stored in "output" directory.

# Memory metrics
In order to assess the performance of the memory-bound PT method, in addition to Tesla V100 GPU, we measured the effective memory throughput metric  ${\bf T}_{eff}$ on Ampere A100 SXM4 featuring 80GB on-board memory. The results are listed below:

| DoFs |  Jakobshavn Isbrae (GB/s)  | DoFs | Pine Island Glacier (GB/s)|       
| :----: | :----: | :----: | :----: | 
| 8e4 | 34 | 3e4 | 17 | 
| 3e5 | 47 | 7e4 | 30 |
| 7e5 | 58 | 1e5 | 36 |
| 1e6 | 56 | 2e6 | 58 |
| 2e7 | 38 | 3e7 | 36 |


# Questions/Comments/Discussion
For questions, comments and discussions please post in the FastIceFlo discussions forum.

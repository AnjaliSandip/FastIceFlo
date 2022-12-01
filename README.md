# PT FEM SSA GPU solver


Copyright (C) 2022 Anjali Sandip, Ludovic Raess and and Mathieu Morlighem.

XXX is a free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

XXX is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with XXX. If not, see http://www.gnu.org/licenses/.

Current version can be found:  here??


- Aim of the project
- Citation information
- Conservation of Momentum Balance equations (+ PT Solver + FEM method)
- Glacier Model Domains/figures
- Generate bin files/steps
- src

- output
- visu
- Additional scripts?
- other related documents
- contact: anjali.sandip@und.edu

# Glacier Model Configurations

To generate the the glacier model configurations and the corresponding DoFs, steps listed below:

![gmd_domain](https://user-images.githubusercontent.com/60862184/204933517-d4b81b5b-acb3-4256-a8be-02439db7f3dc.png)

Figure 1. Glacier model configurations; observed surface velocities interpolated on a uniform mesh. Panels $\textbf{(a)}$ and $\textbf{(b)}$  correspond to Jakobshavn Isbrae and Pine Island Glacier respectively.

Step 1: Install [ISSM](https://issm.jpl.nasa.gov/download/) <br>
Step 2: Modify the average element size or spatial resolution "resol"  <br>

 
| DoFs |  Jakobshavn Isbrae resol (m) | DoFs | Pine Island Glacier resol (m)|       
| :----: | :----: | :----: | :----: | 
| 8e4 | 600 | 3e4 | 2500 | 
| 3e5 | 310 | 7e4 | 1750 |
| 7e5 | 200 | 1e5 | 1250 |
| 1e6 | 170 | 2e6 |  300 |
| 2e7 | ?? | 

Step 3: Run "runme.m" script (located in PIG and JKS folders) <br>
Step 4: To generate the bin file, in the MATLAB command window type
`md=solve(md,'Stressbalance')`

# Running scripts
To perform the numerical experiments described in the pre-print, 
Step 1: clone or download this repository.
Step 2: compile the `ssa_fem_pt.cu` routine on a system hosting an Nvidia Tesla V100 GPU `nvcc -arch=sm_70 -O3 -lineinfo   ssa_fem_pt.cu  -Ddmp=$damp -Dstability=$vel_rela -Drela=$visc_rela`
Step 3: Run it (`./a.out`) 

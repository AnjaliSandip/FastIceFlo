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
 Quickstart:
CUDA > compile the `ssa_fem_pt.cu` routine on a system hosting an Nvidia GPU using the compilation line displayed on the top line of the `.cu` file. Run it (`./a.out`) and use the MATLAB visualisation script to plot
 the output.
- output
- visu
- Additional scripts?
- other related documents
- contact: anjali.sandip@und.edu

# Glacier Model Configurations


![gmd_domain](https://user-images.githubusercontent.com/60862184/204933517-d4b81b5b-acb3-4256-a8be-02439db7f3dc.png)

Figure 1. Glacier model configurations; observed surface velocities interpolated on a uniform mesh. Panels $\textbf{(a)}$ and $\textbf{(b)}$  correspond to Jakobshavn $Isbr{\ae}$ and Pine Island Glacier respectively.


To generate the the glacier model configurations and the corresponding DoFs implemented to assess PT method's performance,

Step 1: Install ISSM <br>
Step 2: Modify the average element size or spatial resolution "resol" corresponding to DoFs for the glacier model configurations <br>

Jakobshavn Isbrae:
| DoFs |  resol (m) |         
| :----: | :----: | 
| 8e4 | 600 | 
| 3e5 | 310 | 
| 7e5 | 200 | 
| 1e6 | 170 | 
| 2e7 | ?? | 


Pine Island Glacier:
| DoFs |  resol (m)| 
| :----: | :----: | 
| 3e4 | 2500 | 
| 7e4 | 1750 | 
| 1e5 | 1250 |
| 2e6 | 300 | 

Step 3: Run runme script (located in PIG and JKS folders) <br>
Step 4: Generate the bin file, in the MATLAB command window type
md=solve(md,'Stressbalance')

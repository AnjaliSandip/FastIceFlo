

We compared the CUDA C PT implementation with the Krylov subspace method relying on biconjugate gradient with block Jacobi pre-conditioner (bcgsl/bjacobi). The chosen CPU architecture was a 64-bit 18-core Intel Xeon Gold 6140 processor with 192 GB of RAM per node. CPU-based multi-core MPI-parallelized ice-sheet flow simulations were executed on two CPU processors, all 36 cores enabled

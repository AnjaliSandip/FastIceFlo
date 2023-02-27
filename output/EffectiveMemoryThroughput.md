In order to assess the performance of the memory-bound PT CUDA C implementation on NVIDIA Tesla V100 SXM2 GPU (16GB DRAM) and a Ampere A100 SXM4 (80GB DRAM) we employ the **effective memory throughput** metric.


|Jakobshavn Isbræ number of vertices | V100 (GB/s) | A100 (GB/s)|
| :----: | :----: | :----: | 
| 44229 | 24 | 34 | 
| 167337 | 23 | 47 |
| 393771 | 23 | 58 |
| 667729 | 19 | 56 |
| 10664257 | 11 | 38 |

|Pine Island Glacier number of vertices | V100 (GB/s) | A100 (GB/s)|
| :----: | :----: | :----: |  
| 35646 | 23 | 30 |
| 69789 | 23 | 36 |
| 1110705 | 18 | 58 |
| 17747649 | 11 | 36 |

We compared the effective memory throughput with the peak, i.e. the memory transfer speed for performing memory copy operations only. It represents the hardware performance limit. The code [`memcopy.cu`](/scripts/memcopy.cu) located in the [scripts](/scripts) directory reports the peak memory throughput. The reported peak memory throughput for the GPU hardware NVIDIA Tesla V100 and NVIDIA A100 were 785 GB/s and 1536 GB/s respectively.  

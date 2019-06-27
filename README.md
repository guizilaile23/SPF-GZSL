# SPF-GZSL
---------------------------------------------------------------------------------------------------
Code of paper "Generalized Zero Shot Learning via Synthesis Pseudo Features"
---------------------------------------------------------------------------------------------------
  
# Requirement
Python > 3.6  
Pytorch > 1.0.0  
Cuda  
# Data
Download data from [here](https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) and unzip it `unzip data.zip`. The data used in this paper is provided by Xian(https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/)



# Result
GZSL performance evaluated under the setting proposed in [Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly](https://arxiv.org/abs/1707.00600).Yongqin Xian, Christoph H. Lampert, Bernt Schiele, Zeynep Akata.  
ResNet-101 feature, GBU split, averaged per class accuracy.  

| Model      |    AWA1 ts    |    AWA1 tr |    AWA1 H    |    AWA2 ts    |    AWA2 tr  |   AWA2 H    |    CUB ts  |   CUB tr |   CUB H    |   SUN ts   |   SUN tr    |   SUN H    |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| DEM   |   32.8   |   84.7   |   47.3   |   30.5	  |   86.4	 |   45.1   |   19.6	 |   54.0	  |   13.6	 |   20.5  |	34.3	 |  25.6  | 
| LESAE |   19.1	 |   70.2	  |   30.0	 |   21.8	  |  70.6	  |  33.3	  |  24.3	  |  53.0	  |  33.3	  |  21.9	  |  34.7	  |  26.9| 
| TVN   |   27.0	  |  67.9	  |  38.6	  |  –	  |  –	  |  –	  |  26.5	  |  62.3	  |  37.2	  |  22.2	  |  38.3	  |  28.1 |
| ZSKL  |   18.3	  |  79.3	  |  29.8	  |  18.9	  |  82.7	  |  30.8	  |  24.2	  |  63.9	  |  35.1	  |  21.0	  |  31.0	  |  25.1 |
| CSSD  |   34.7	  |  87.1	  |  49.6	  |  –	  |  –	  |  –	  |  19.1	  |  62.7	  |  29.3	  |  –	  |  –	  |  – |
| BZSL  |   19.9	  |  23.9	  |  21.7	  |  –	  |  –	  |  –	  |  18.9	  |  25.1	  |  20.9	  |  17.3  |  	17.6	  |  17.4 |
| UVDS  |   15.3	  |  79.5	  |  25.7	  |  –	  |  –	  |  –	  |  23.8	  |**76.5** |  36.3	  |  –	  |  –	  |  – |
| DCN   |   25.5	  |  84.2	  |  39.1	  |  –	  |  –	  |  –	  |  28.4	  |  60.7	  |  38.7	  |  25.5	  |  37.0	  |  30.2 |
| NIWT  |   –	  |  –  |  	–  |  	42.3	  |  38.8	  |  40.5	  |  20.7	  |  41.8	  |  27.7	  |  –	  |  –	  |  – |
| RN    |   31.4	  |**91.3**|  46.7	  |  30.0	  |**93.4**|  45.3	  |**38.1** |  61.1	  |**47.0**|  –	  |  –	  |  – |
| ours  | **48.5** |   59.8   | **53.6** | **52.4**|   60.9   |**56.3**|   30.2   |   63.4   |   40.9   |**32.2** |**59.0** |**41.6** | 

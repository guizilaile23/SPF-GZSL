# SPF-GZSL
---------------------------------------------------------------------------------------------------
Code of paper "Generalized Zero Shot Learning via Synthesis Pseudo Features"
---------------------------------------------------------------------------------------------------
The code will be released after the paper is accepted.
and for now, I uploaded the train and test log file first.  
# Requirement
Python > 3.6  
Pytorch > 1.0.0  
Cuda  
# Data
Download data from [here](https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) and unzip it `unzip data.zip`.

# Result
GZSL performance evaluated under the setting proposed in [Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly](https://arxiv.org/abs/1707.00600).Yongqin Xian, Christoph H. Lampert, Bernt Schiele, Zeynep Akata.  
ResNet-101 feature, GBU split, averaged per class accuracy.  

| Model      |    AWA1 ts    |    AWA1 tr |    AWA1 H    |    AWA2 ts    |    AWA2 tr  |   AWA2 H    |    CUB ts  |   CUB tr |   CUB H    |   SUN ts   |   SUN tr    |   SUN H    |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| ours       |   48.5   |   59.8   |   53.6   |   52.4   |   60.9   |   56.3   |   30.2   |   63.4   |   40.9   |**32.2** |**59.0** |**41.6** |  

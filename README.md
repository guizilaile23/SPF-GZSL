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
Download data from [here](http://www.robots.ox.ac.uk/~lz/DEM_cvpr2017/data.zip) and unzip it `unzip data.zip`.


# Result
GZSL performance evaluated under the setting proposed in [Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly](https://arxiv.org/abs/1707.00600).Yongqin Xian, Christoph H. Lampert, Bernt Schiele, Zeynep Akata.

ResNet-101 feature, GBU split, averaged per class accuracy.
|            |     AWA1 T1     |      AWA2 T1         |      CUB T1         |    SUN T1     |   
| Model      |  u  |  s  |  H  |  u  |  s  |  H  |  u  |  s  |  H  |  u  |  s  |  H  |  
|------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|  
| ours       |48.5 |59.8 |53.6 |52.4 |60.9 |56.3 |30.2 |63.4 |40.9 |**32.2** |**59.0** |**41.6** |  

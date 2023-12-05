# DIEM-SR
This is an offical implementation of the TCSVT2023's paper [Dynamic Degradation Intensity Estimation for Adaptive Blind Super-Resolution: A Novel Approach and Benchmark Dataset](https://ieeexplore.ieee.org/document/10314558).

If you find this repo useful for your work, please cite our paper:
```
@ARTICLE{10314558,
  author={Chen, Guang-Yong and Weng, Wu-Ding and Su, Jian-Nan and Gan, Min and Chen, C. L. Philip},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Dynamic Degradation Intensity Estimation for Adaptive Blind Super-Resolution: A Novel Approach and Benchmark Dataset}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2023.3331883}}
```

The codes are built on the basis of [BasicSR](https://github.com/xinntao/BasicSR).

## Dependences
1. lpips (pip install --user lpips)
2. matlab (to support the evaluation of NIQE). The details about installing a matlab API for python can refer to [here](https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

## Datasets
The datasets DIV2K can be downloaded from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/). And the datasets Real-BlindSR dataset can be downloaded from [here](http://ieee-dataport.org/documents/real-blindsr-dataset). 

## Start up
To get a quick start:

```bash
cd codes/config/DIEM-SR/
python3 test.py --opt options/test/test_DIEM-SR.yml
```

# DIEM-SR

The codes are built on the basis of [BasicSR](https://github.com/xinntao/BasicSR).

## Dependences
1. lpips (pip install --user lpips)
2. matlab (to support the evaluation of NIQE). The details about installing a matlab API for python can refer to [here](https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

## Datasets
The datasets DIV2K can be downloaded from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/). And the datasets Real-BlindSR dataset can be downloaded from [here](10.21227/z2yz-c641). 

## Start up
To get a quick start:

```bash
cd codes/config/DIEM-SR/
python3 test.py --opt options/test/test.yml
```

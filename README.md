## Requirements
- **Python 3.8**
- **Windows10, Ubuntu18.04 or higher**
- **NVDIA GeForce RTX 3090**
- **pytorch 1.8.0 or higher**

## Datasets

**We used the NUDT-SIRST, IRSTD-1K and sirst-aug for both training and test. Three datasets can be found and downloaded in: [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection), [IRSTD-1K](https://github.com/RuiZhang97/ISNet), [SIRST-Aug](https://github.com/Tianfang-Zhang/AGPCNet). Or you can just download our collated dataset in [Baidu Driver](https://pan.baidu.com/s/1d6nboVuv_YO19Svd2GHNng)(key:x5ij )**

**Please first download these datasets and place the 3 datasets to the folder `./data/`. More results will be released soon!** 



* **Our project has the following structure:**
```
├──./data/
│    ├── NUDT-SIRST
│    │    ├── images
│    │    │    ├── 000001.png
│    │    │    ├── 000002.png
│    │    │    ├── ...
│    │    ├── img_idx
│    │    │    ├── test_NUDT-SIRST.txt
│    │    │    ├── train_NUDT-SIRST.txt
│    │    ├── masks
│    │    │    ├── 000001.png
│    │    │    ├── 000002.png
│    │    │    ├── ...
│    ├── IRSTD-1K
│    │    ├── images
│    │    │    ├── XDU0.png
│    │    │    ├── XDU1.png
│    │    │    ├── ...
│    │    ├── img_idx
│    │    │    ├── test_IRSTD-1K.txt
│    │    │    ├── train_IRSTD-1K.txt
│    │    ├── masks
│    │    │    ├── XDU0.png
│    │    │    ├── XDU1.png
│    │    │    ├── ...
│    ├── sirst_aug
│    │    ├── images
│    │    │    ├── 000000.png
│    │    │    ├── 000001.png
│    │    │    ├── ...
│    │    ├── img_idx
│    │    │    ├── test.txt
│    │    │    ├── train.txt
│    │    ├── masks
│    │    │    ├── 000000_mask.png
│    │    │    ├── 000001_mask.png
│    │    │    ├── ...
```

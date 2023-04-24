#### This repo is for the methods described in the paper.
Wang QH, Ho JY, Huang TY, “Open-source segmentation toolbox for 4D Cardiac MRI: Enhancing U-Net Performance through Spatial-Temporal Features and Edge Labeling” 

* This repo is only for CPU-based segmentation.
* For updated models , please visit: https://github.com/htylab/tigerhx



## Background

* This repo provides deep-learning methods for cardiac MRI segmentation
### Usage
```
    python comb.py c:\data\*.nii.gz -o c:\output
    python comb.py c:\data\subject.nii.gz -o c:\output
    python comb.py c:\data\subject.nii.gz
    python comb.py c:\data\*.nii.gz -o c:\output -a # producing AHA segments with this option
```
* Please note that upon first use, the program will require an internet connection to download the model file.
## Citation

* If you use this application, cite the following paper:

1. Open-source segmentation toolbox for 4D Cardiac MRI: Enhancing U-Net Performance through Spatial-Temporal Features and Edge Labeling


#### We gratefully acknowledge the authors of the U-Net implementation:
https://github.com/ellisdg/3DUnetCNN

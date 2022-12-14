# DeepFusion: a deep bi-modal information fusion network for unraveling protein-RNA interactions using in vivo RNA structures

Welcome to DeepFusion's git repository.

# Installation
DeepFusion is designed for the Linux operating system environment. All code is built on the Python programming language and the Pytorch deep learning framework. Please confirm the following settings progressively: <br>
```
Python 3.6.9  
pip 9.0.1 
Pytorch 1.2.0 
pip3 install sklearn 
pip3 install pandas 
pip3 install tqdm 
```
***
# Dataset
DeepFusion supports both RBP-24 and RBP-120 datasets. RBP-120 can be downloaded from the URL. This repository contains sample data from the RBP-120 dataset in three sets of experiments: sequence-only, sequence+vitro, and sequence+vivo, in the folders **_/RBP-120/sequence-only/sample_**, **_/RBP-120/sequence+vitro/sample_** and **_/RBP-120/sequence+vivo/sample_**.
***
# usage
Documentation for DeepFusion on **RBP-24** and **RBP-120** is available in the **_/RBP-24_** and **_/RBP-120_** folders, respectively. 
## Testing
To reproduce the DeepFusion results, please run the ```DeepFusion_test.py``` file.
```
python3 DeepFusion_test.py RBP-120
```
The 'RBP-120' parameter can be replaced with the 'RBP-24'.
## Training
To train your own model on your own server, please run the ```DeepFusion_train.py``` file.
```
python3 DeepFusion_train.py RBP-120 sequence
```
Parameter#2 refers to the data set (RBP-24 or RBP-120). Parameter#3 refers to the experiment (sequence, vitro, or vivo).
## Visualization
For visualization of the features extracted from DeepFusion, follow the steps below to modify the code to suit your server's format and use ```/RBP-120/motif/gen_featuremap_120.py```. The tomtom file generated by the code can be used for TOMTOM platform plotting. Same for other experiments.

+ **os.environ['CUDA_VISIBLE_DEVICES']**: Specification of the GPU number.
+ **rbpnames**: List of RBPs used for visualisation.
+ **data_dir**: The path of the dataset used for visualisation.
## Tips
If you want to run the DeepFusion model on your own dataset, please process the data into ['label', '75 nucleotides', '375 nucleotides'] standard format, as shown in **_/RBP-120/sequence-only/sample_**. If you find problems with the code running please first check the following parameters???

+ **os.environ['CUDA_VISIBLE_DEVICES']**: Specification of the GPU number used for training.
+ **result_path**: The path where the training results are saved.
+ **name_list**: The path to all processed RBP-120 sequence-only data.
***
## License
This project is free to use for non-commercial purposes - see the [LICENSE](https://github.com/Qiaoyx97/DeepFusion/blob/main/LICENSE) file for details.
***
## Contact
For any questions, please contact biozy@ict.ac.cn.


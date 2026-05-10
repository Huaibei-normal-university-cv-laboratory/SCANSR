# SCANSR
Our paper is currently under submission, and detailed data information will be made public after its publication

## Dependencies
- Python 3.8
- torch 1.10.1
- torchaudio 0.10.1
- torchvision 0.11.2
- numpy 1.24.4
- PyYAML 6.0.2
- opencv 4.10.0
- pillow 10.4.0
- monai 1.3.2
- timm 1.0.11

## Train
### 1. Prepare training data
Download Brats2018 dataset and IXI dataset.  

process dataset by processDataset.py (data_tools/).
### 2. Begin to train
python train.py -v "version" -p train --train_yaml "xxx.yaml"

## Quick Test 
python test.py -v "version" -s 153 -t tester_Matlab --test_dataset_name "dataset"# SCANSR

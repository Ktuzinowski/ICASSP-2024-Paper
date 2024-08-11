# ICASSP-2024-Paper
Repository for scripts used to generate figures for ICASSP conference. Code for paper that is looking to get published.

## ImageNet Faster Download
1. Download tar files containing both training and validation datasets for ImageNet.
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
```
1.1. Alternative to the parallel quicker approach outlined below is just using a single script to load in training and validation data. Slow compared to the Python files listed below.
```
./imagenet_download_scripts/extract_ILSVRC.sh
```
2. Run Python file **extract_ILSVRC_training_data_from_ZIP.py** in the same directory as **ILSVRC2012_img_train.tar**. Should create a folder **imagenet/train**.
```
python imagenet_download_scripts/extract_ILSVRC_training_data_from_ZIP.py
``
3. Run Python file **extract_ILSVRC_validation_data_from_ZIP.py** in the same directory as **ILSVRC2012_img_val.tar**. Should create a folder **imagenet/val**.
```
python imagenet_download_scripts/extract_ILSVRC_validation_data_from_ZIP.py
```
4. Create folders for validation data (class folders) using script **create_folders_for_validation_data.sh**.
```
./imagenet_download_scripts/create_folders_for_validation_data.sh
```
5. Test if dataset currently works with test script **test_utilize_imagenet.py**.
```
python imagenet_download_scripts/test_utilize_imagenet.py
```
## Vision Transformer Models
* Inside the folder **vit**, there is a Python file **model_vit.py**. Inside this is a way to get the ViT attention output weights for each encoder block. Use this architecture for training ViTs to compare Mean Attention Distance.

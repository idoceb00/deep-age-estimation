
# Deep AgeNet: Facial Age Estimation via Transfer Learning with ResNet
This repository contains the implementation of DeepAgeNet, a deep learning model for age estimation based on ResNet-50.

## Dependencies (conda)

To install all required packages using conda:

1. Create the environment:

    conda env create -f environment.yml

2. Activate it:

    conda activate deep-age-estimation

This will install:

- Python 3.10
- PyTorch ≥ 2.0
- torchvision
- pandas
- scikit-learn
- matplotlib
- Pillow


## How to Run
1. Place the datasets inside the `data/` directory, following the expected structure.
2. Select the dataset to use by modifying the corresponding loader in `main.py`.
3. Run the training and evaluation:

    python main.py

Results (MAE, MSE, R², Accuracy@±5) and residual plots will be saved automatically.

## Datasets (GDrive Links)
- CACD: https://drive.google.com/file/d/1j4tkc7PJ2E3rtAwr2kk1Gz2CP2RBmxOz/view?usp=drive_link
- FG-NET: https://drive.google.com/file/d/1Ms2zW7yVcDcZ2b8AclokitvwGek69QgU/view?usp=drive_link
- MORPH: https://drive.google.com/file/d/1Jmff9wzPwnxiNH5nLdJWkmKVpei6asfN/view?usp=drive_link
- AgeDB: https://drive.google.com/file/d/1wETT9U1sjgeKwQ9PdODxtk1zGQUrXjYh/view?usp=drive_link
- UTKFace: https://drive.google.com/file/d/182JDjfvNggwjuEyuDkB9E7KKn1L6THPm/view?usp=drive_link
- IMDB-WIKI: https://drive.google.com/file/d/1Hs1xMPW8-mwnS-79BbNfDVQwm_AFVHmj/view?usp=drive_link
- MegaAge: https://drive.google.com/file/d/1vT2rXTMAgQCTNY8oFZMY_1Brc0u8ZWuf/view?usp=drive_link
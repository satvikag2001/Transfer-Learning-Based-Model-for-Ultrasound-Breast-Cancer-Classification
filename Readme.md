# Transfer Learning Based Model for Breast Cancer Classification

This repository contains an implementation associated with the paper *A Novel Transfer Learning-Based Model for Ultrasound Breast Cancer Image Classification* (Gupta, Agrawal, Singh, and Kumar, 2023). The work applies transfer learning to classify breast cancer images and evaluates a transfer-learning feature representation with a set of classifiers.

Paper: https://link.springer.com/chapter/10.1007/978-981-19-9819-5_37

## Overview

The published paper proposes a transfer learning model for classifying breast ultrasound images as normal, benign, or malignant. The model is built on a modified ResNet50 network that is pre-trained on the ImageNet dataset and then trained further on the Breast Ultrasound Images (BUSI) dataset. Custom layers are added at the head of the network to extract features from the ultrasound images. The authors report 97.8% accuracy, 97.68% recall, 99.21% precision, and 98.44% F1-score for breast cancer detection.

The scripts committed to this repository implement a related transfer learning workflow on the BreaKHis 400X breast histopathology dataset. A VGG19 network serves as the feature extractor and classifier, and a separate script evaluates a feature table with several standard machine learning models. The two scripts are described below.

## Repository Contents

| File | Description |
| --- | --- |
| `preprocess.py` | Loads a saved Keras model and evaluates it on the BreaKHis 400X test images for benign and malignant classes. |
| `classification.py` | Trains and evaluates several classifiers on a feature table exported to CSV. |
| `breakhis400x.csv` | Feature table used by `classification.py`. The first column holds the class label and the remaining columns hold the extracted features. |

## How It Works

### preprocess.py

This script performs image-level evaluation of a trained model.

1. It loads a pre-trained Keras model from `vgg19-90.6.h5`.
2. It reads the test images for the benign and malignant classes from the BreaKHis 400X directory structure.
3. Each image is resized to 224 x 224 x 3, converted to an array, reshaped to a batch of one, and scaled to the range 0 to 1.
4. The model predicts a score for each image. A score below 0.5 is treated as benign and a score above 0.5 is treated as malignant.
5. The script prints the overall classification accuracy across the test set.

### classification.py

This script evaluates the exported feature representation with a set of classifiers.

1. It reads `breakhis400x.csv` and separates the label column from the feature columns.
2. It splits the data into training and test sets, with 20 percent held out for testing.
3. It trains and reports accuracy for the following models:
   - Random Forest
   - Logistic Regression
   - A feedforward neural network built with Keras, with dense layers of 512 and 256 units using ReLU activation and a single sigmoid output unit, trained for 150 epochs with the Adam optimizer and binary cross-entropy loss
   - Gaussian Naive Bayes
   - K-Nearest Neighbors, with the number of neighbors swept from 3 to 29

## Requirements

- Python 3
- numpy
- pandas
- scikit-learn
- tensorflow

Install the dependencies with pip:

```bash
pip install numpy pandas scikit-learn tensorflow
```

## Additional Files Required

The scripts reference two assets that are not included in this repository. You need to provide them before running the code.

1. `vgg19-90.6.h5`, the trained Keras model loaded by `preprocess.py`. Place this file in the repository root.
2. The BreaKHis 400X image dataset, organized so that the test images are available under `BreaKHis 400X/BreaKHis 400X/BreaKHis 400X/test/benign` and `BreaKHis 400X/BreaKHis 400X/BreaKHis 400X/test/malignant`, matching the paths used in `preprocess.py`.

The CSV path inside `classification.py` is `Breakhis400x.csv`. Confirm that the filename casing matches the file in your environment.

## Usage

Evaluate the trained model on the test images:

```bash
python preprocess.py
```

Train and compare the classifiers on the feature table:

```bash
python classification.py
```

Each script prints accuracy values to the console.

## Dataset

- BUSI is the Breast Ultrasound Images dataset used in the published paper. Reference: Al-Dhabyani, W., Gomaa, M., Khaled, H., Fahmy, A. Dataset of breast ultrasound images. Data in Brief, 28, 104863 (2020).
- BreaKHis 400X is a subset of the Breast Cancer Histopathological Image dataset at 400x magnification, used by the scripts in this repository.

## Citation

If you use this work, please cite the paper:

```bibtex
@inproceedings{gupta2023transferlearning,
  title     = {A Novel Transfer Learning-Based Model for Ultrasound Breast Cancer Image Classification},
  author    = {Gupta, Saksham and Agrawal, Satvik and Singh, Sunil K. and Kumar, Sudhakar},
  booktitle = {Computational Vision and Bio-Inspired Computing},
  series    = {Advances in Intelligent Systems and Computing},
  volume    = {1439},
  pages     = {511--523},
  year      = {2023},
  publisher = {Springer, Singapore},
  doi       = {10.1007/978-981-19-9819-5_37}
}
```

## License

No license file is included in this repository. Contact the repository owner for terms of use.

## Notes

The published paper describes a ResNet50 model trained on the BUSI ultrasound dataset for three-class classification. The scripts in this repository use a VGG19 backbone and the BreaKHis 400X histopathology dataset for binary classification of benign and malignant samples. Read both the paper and the code to understand the scope of each.

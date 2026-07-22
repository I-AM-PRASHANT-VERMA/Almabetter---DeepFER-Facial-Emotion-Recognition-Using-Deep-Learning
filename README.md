![DeepFER banner](assets/deepfer-banner.png)

# DeepFER: Facial Emotion Recognition Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-FF6F00?logo=tensorflow&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU-F9AB00?logo=googlecolab&logoColor=white)

DeepFER is an individual deep learning project that classifies facial images into seven emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

## Overview

The notebook builds an end-to-end facial emotion recognition workflow using seven classes: angry, disgust, fear, happy, neutral, sad, and surprise. It includes data-quality checks, duplicate handling, preprocessing, augmentation, CNN training, transfer-learning comparison, evaluation, explainability, and saved-model testing.

## Final Result

| Selected model | Leakage-safe accuracy | Leakage-safe macro F1 | Official validation accuracy | Official validation macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Reference-style FER CNN | 0.654 | 0.616 | 0.666 | 0.641 |

The final CNN was selected because its macro F1-score gives a more balanced comparison across all seven emotions. It performed better than the baseline CNN and the EfficientNetB0 transfer-learning comparison.

## Run The Project

1. Open [`DeepFER_Facial_Emotion_Recognition.ipynb`](DeepFER_Facial_Emotion_Recognition.ipynb) in Google Colab.
2. Enable a GPU runtime.
3. Keep `Face Emotion Recognition Dataset.zip` in the project Drive folder.
4. Run the notebook from top to bottom.

The dataset zip is 120.5 MB, so it is stored in the project Google Drive folder rather than GitHub. The final trained model is also stored in Drive as `reference_fer_cnn_final.keras`.

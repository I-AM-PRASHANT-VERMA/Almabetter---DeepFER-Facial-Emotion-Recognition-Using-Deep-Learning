![DeepFER banner](assets/deepfer-banner.png)

# DeepFER: Facial Emotion Recognition Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-FF6F00?logo=tensorflow&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU-F9AB00?logo=googlecolab&logoColor=white)

DeepFER is an individual deep learning project that classifies facial images into seven emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

## Overview

The notebook builds an end-to-end facial emotion recognition workflow using seven classes: angry, disgust, fear, happy, neutral, sad, and surprise. It includes data-quality checks, duplicate handling, preprocessing, augmentation, CNN training, transfer-learning comparison, evaluation, explainability, and saved-model testing.

## Evaluation Protocol

Exact duplicate hashes with conflicting labels are removed first. The cleaned training pool is split stratified by emotion into model-training and tuning-validation sets. Model checkpoints and model selection use tuning macro F1-score only.

The duplicate-safe official validation images are kept as the held-out final evaluation set. After model selection, only the selected model is evaluated on this set. The saved model, results CSV, and JSON metadata all come from the same completed run.

The reference CNN achieved the best tuning result with 65.4% accuracy and 0.625 macro F1. On the held-out final set, it achieved 65.9% accuracy, 0.615 macro F1, and 0.654 weighted F1. Happy was the strongest class and fear remained the weakest.

## Limitations

This is a classroom image-classification prototype trained on small 48 by 48 grayscale face crops. It is not validated for clinical, mental-health, hiring, security, or emotion-profiling decisions. The dataset has strong class imbalance and no person identifier, so near-duplicate or same-person overlap cannot be ruled out.

## Run The Project

1. Open [`DeepFER_Facial_Emotion_Recognition.ipynb`](DeepFER_Facial_Emotion_Recognition.ipynb) in Google Colab.
2. Enable a GPU runtime.
3. Keep `Face Emotion Recognition Dataset.zip` in the project Drive folder.
4. Run the notebook from top to bottom.

The dataset zip is 120.5 MB, so it is stored in the project Google Drive folder rather than GitHub. A clean run writes the selected `.keras` model, `deepfer_model_results.csv`, and `deepfer_run_metadata.json` to the Drive `outputs` folder.

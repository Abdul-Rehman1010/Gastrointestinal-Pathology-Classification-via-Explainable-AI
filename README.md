# Explainable AI for Gastrointestinal Image Classification (Kvasir Dataset)

This repository contains an end-to-end deep learning pipeline for classifying gastrointestinal track images using the Kvasir-v2 dataset. It integrates Explainable AI (XAI) using LIME (Local Interpretable Model-agnostic Explanations) to provide transparent, interpretable predictions for medical image analysis.

## Overview

Medical image classification requires high accuracy and interpretability. A "black box" model is often insufficient for clinical settings. This project leverages a fine-tuned **EfficientNetB3** model for robust feature extraction and classification across 8 distinct classes of gastrointestinal anomalies and landmarks. 

To build trust in the model's predictions, **LIME** is integrated to visually highlight the superpixels (regions) in the endoscopy images that contributed most significantly to the model's final decision.

## Dataset

The project uses the **Kvasir Dataset v2**, which consists of images from inside the gastrointestinal (GI) tract. 

The data is classified into the following 8 classes:
* `dyed-lifted-polyps`
* `dyed-resection-margins`
* `esophagitis`
* `normal-cecum`
* `normal-pylorus`
* `normal-z-line`
* `polyps`
* `ulcerative-colitis`

## Model Architecture

* **Base Model:** EfficientNetB3 (pre-trained on ImageNet).
* **Custom Classifier Head:**
    * Batch Normalization
    * Dense Layer (256 units, ReLU, L2 Regularization)
    * Dropout (0.4)
    * Output Dense Layer (8 units, Softmax)
* **Optimizer:** Adamax (Learning Rate: 0.001)
* **Callbacks:** Early Stopping, ReduceLROnPlateau, ModelCheckpoint

## Explainable AI (XAI) Integration

**LIME** is utilized to explain individual predictions. By perturbing the input image and observing the changes in the model's output, LIME builds a local linear model around the prediction. The output is a visual mask overlaid on the original image, highlighting the specific anatomical features or anomalies that drove the classification.

## Repository Structure

```text
├── explainable_ai_kvasir.ipynb 
├── class_indices.json          
├── index_to_class.json         
├── final_efficientnetb3_kvasir_model.keras 
└── README.md

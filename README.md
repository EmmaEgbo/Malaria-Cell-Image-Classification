# Malaria Cell Image Classifier with Transfer Learning

## Project Overview

This project develops a deep learning model to classify red blood cell images as either **infected** or **uninfected**, using the **NIH Malaria Cell Images Dataset**.
The goal is to leverage **transfer learning** to build an accurate and efficient classifier that can aid early malaria detection and diagnosis.

---

## Dataset

The model uses the **NIH Malaria Cell Images Dataset**, accessed via **TensorFlow Datasets**.
It contains approximately **27,558 labeled images** divided into two categories:

* **0 – Parasitized**
* **1 – Uninfected**

The dataset is split into **training (80%)**, **validation (10%)**, and **testing (10%)** subsets.

---

## Data Preprocessing

* All images are resized to **(224 × 224)** pixels.
* Pixel values are normalized to the range **[0, 1]**.
* The datasets are batched and prefetched for optimized GPU/TPU performance.

---

## Model Architecture

The project employs **Transfer Learning** using **MobileNetV2**, pre-trained on **ImageNet**, as the base model.

* The base model’s layers are **frozen** initially to preserve pre-learned features.
* Added layers:

  * **Global Average Pooling**
  * **Dense output layer** with **sigmoid** activation for binary classification.
* The model is compiled using:

  * **Optimizer:** Adam
  * **Loss:** Binary Cross-Entropy
  * **Metrics:** Accuracy

---

## Training and Fine-Tuning

1. **Initial Training:**
   The model is trained for a few epochs with the base layers frozen.
2. **Fine-Tuning:**
   Selected layers of the base model are unfrozen and retrained with a **lower learning rate** to adapt the model to the malaria dataset.

---

## Evaluation

Model performance is evaluated using standard metrics:

* Accuracy
* Precision, Recall, F1-Score
* Confusion Matrix
* ROC Curve and AUC

---

## Dependencies

* TensorFlow
* TensorFlow Datasets
* NumPy
* Matplotlib
* Scikit-learn
* Seaborn

Install all dependencies with:

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib scikit-learn seaborn
```

---

## How to Use

You can run this project locally in **Jupyter Notebook** or on **Google Colab**.

1. Clone the repository:

   ```bash
   git clone https://github.com/EmmaEgbo/Malaria-Cell-Image-Classification.git
   cd Malaria-Cell-Image-Classification
   ```
2. Install dependencies.
3. Open the notebook:

   ```bash
   jupyter notebook Malaria_Cell_Image_Classifier.ipynb
   ```
4. Run all cells sequentially to train, fine-tune, and evaluate the model.
5. Save or export your trained model for deployment.

---

## Saved Model

The trained model is saved as:
`malaria_tl_model.keras`
This can be loaded later for inference or integration into applications.

# HEAL-OS-Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Paper](https://img.shields.io/badge/Paper-BMC%202025-blue)](https://doi.org/10.1186/s12911-025-02998-6)

This repository contains the official implementation of the paper:

**"Hierarchical Embedding Attention for Overall Survival Prediction in Lung Cancer from Unstructured EHRs"**  
Authors: Domenico Paolo, Carlo Greco, Alessio Cortellini, Sara Ramella, Paolo Soda,  
Alessandro Bria, Rosa Sicilia.

---

## 🔍 Overview

HEAL is an interpretable deep learning framework for **prognosis prediction** from **unstructured EHRs**.  
It integrates:

- A **multiclass NER** system trained on 25 lung-cancer–specific entity types.
- A **hierarchical attention mechanism** that aggregates entity embeddings at:
  - token level  
  - sentence level  
- A **DeepHit-based network** for survival prediction.

The method significantly outperforms manually extracted clinical features and baseline models.

---

## 📦 Repository Structure
```
HEAL-OS-Prediction/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   ├── networks.py       # hierarchical attention + DeepHit architecture
│   ├── datasets.py       # NER embeddings
│   ├── import_data.py
│   ├── losses.py         # DeepHit Loss
│   ├── main_clinical.py  # train the model based on manually-extracted clinical features
│   ├── main.py           # train HEAL
│   ├── utils_data.py
│   ├── utils_model.py
│   ├── utils_eval.py
│   ├── utils_network.py
│
├── figures/
    ├── Proposed Approach.pdf
```

## 🧠 Method

The workflow includes:
	1.	NER fine-tuning
Using the MedBITR3+ checkpoint (BioBIT-based) with focal loss for class imbalance.
	2.	Embedding extraction
We extract the contextualized embeddings of entity tokens from the NER transformer.
	3.	Hierarchical Attention (HEAL)
	•	Token-level attention
	•	Sentence-level attention
	•	Shared attention weights
	4.	Survival prediction
DeepHit model with calibration, ranking, and likelihood losses.

## 💾 Data

The clinical reports used in this study are not publicly distributable due to patient privacy restrictions.

The NER system used in this work is available in a dedicated repository:
[Italian-NSCLC-NER](https://github.com/nico9902/Italian-NSCLC-NER).

## 🚀 How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
  ```
2. Train the NER model and extract embeddings: use
3. Train the HEAL model using random search to optimize hyperparameters:
   ```
  python -m scripts.main \
    --random_search \
    --rs_iteration 100 \
    --end_fold 10 \
    --attention_mode hierarchical \
    --embedding_path "<path>" \
    --prediction_path "<path>" \
    --label_path "<path>" \
    --kfold_path "<path>" \
    --survival_file_path "<path-to-survival.xlsx>"
  ```

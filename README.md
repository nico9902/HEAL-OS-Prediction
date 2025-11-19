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
    ├── proposed_method.pdf

# HEAL-OS-Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Paper]([https://img.shields.io/badge/Paper-IEEE%202023-blue](https://link.springer.com/article/10.1186/s12911-025-02998-6))]( https://doi.org/10.1186/s12911-025-02998-6)

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

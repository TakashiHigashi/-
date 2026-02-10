# Research Achievements
This repository compiles research activities conducted during the master's program.

# Enhancing Computational Efficiency in Video Action Recognition via Temporal Token Merging in ViViT

This repository provides the official PyTorch implementation of  
"Enhancing Computational Efficiency in Video Action Recognition via Temporal Token Merging in ViViT".

## Overview
We propose a temporal token merging method for ViViT to reduce computational cost
while maintaining recognition accuracy.

## Environment
- Python 3.10
- PyTorch 2.1
- CUDA 11.8

## Dataset
We use the NTU RGB+D dataset.
Please download it from the official website and place it as follows:
https://rose1.ntu.edu.sg/dataset/actionRecognition/

## Correspondence to the Paper

- **Section 3.1 (Proposed MAADS Architecture)**  
  - `MAADS.py`  
  Implementation of the proposed MAADS model and its core adaptive attention mechanism.

- **Section 3.2 (Transformer Backbone)**  
  - `vivit.py`  
  - `custom_vit.py`  
  ViViT backbone and customized Vision Transformer implementations used in MAADS.

- **Section 3.3 (Data Preparation and Input Pipeline)**  
  - `testavi.py`  
  - `huriwake.py`  
  Video data loading, preprocessing, and train/test split procedures.

- **Section 4 (Experiments and Evaluation)**  
  - `maadstraining.py`  
  Training script corresponding to the experimental setup described in the paper.

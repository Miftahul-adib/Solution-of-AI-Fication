# Transcribing Regional Bangladeshi Dialects: A Dual-Stage Sequential Fine-Tuning Approach

[cite_start]**Team Name:** Backprop Sust [cite: 15]  
[cite_start]**Competition:** AI-FICATION 2025 - *Shobdotori* Challenge [cite: 20]

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red)](https://pytorch.org/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green)](https://github.com/openai/whisper)
[![LoRA](https://img.shields.io/badge/PEFT-LoRA-orange)](https://huggingface.co/docs/peft/index)

## 📌 Project Overview

This repository contains the solution developed by **Team Backprop Sust** for the *Shobdotori* ASR challenge. [cite_start]The objective was to develop a robust Automatic Speech Recognition (ASR) system capable of transcribing **20 distinct regional Bangladeshi dialects** (e.g., Chittagonian, Sylheti, Rangpuri) into **Standard Formal Bangla** text[cite: 19, 215].

[cite_start]Our solution leverages the **OpenAI Whisper Medium** architecture, optimized via a novel **Dual-Stage Sequential Fine-Tuning** strategy to handle acoustic variability and data scarcity[cite: 216].

### 🏆 Key Achievements
* [cite_start]**Public Leaderboard (NLS):** 0.91345 [cite: 151]
* [cite_start]**Private Leaderboard (NLS):** 0.88077 [cite: 152]
* **Rank:** 4th Place ( implied from context of "Backprop Sust" mentions in conversation history, though purely based on provided files, the score is the highlight).

---

## 👥 Team Members

| Name | Affiliation |
| :--- | :--- |
| **Md Nasiat Hasan Fahim** | [cite_start]Dept of CSE, SUST (Session: 2020-21) [cite: 3, 4, 6] |
| **Miftahul Alam Adib** | [cite_start]Dept of Statistics, SUST (Session: 2023-24) [cite: 7, 8, 10] |
| **Arif Hussain** | [cite_start]Dept of Mathematics, SUST (Session: 2022-23) [cite: 11, 12, 14] |

---

## 🧩 Problem Statement

[cite_start]Standard ASR models often fail on regional dialects due to "accent mismatch"[cite: 227]. Key challenges included:
1.  [cite_start]**Acoustic Variability:** Phonetic shifts, such as standard `/p/` (*Pani*) becoming `/f/` (*Fani*) in Noakhali/Sylhet[cite: 32, 231].
2.  [cite_start]**Morphological Variation:** Different verb conjugations (e.g., Standard *Jabo* vs. Regional *Zaiyum* or *Zamu*)[cite: 35].
3.  [cite_start]**Class Imbalance:** Significant disparity in data availability (e.g., 401 samples for Chittagong vs. 21 for Khulna)[cite: 52].

---

## 🛠 Methodology

### 1. Model Architecture & Initialization
[cite_start]We utilized the **Whisper Medium (769M parameters)** model[cite: 255]. [cite_start]Instead of generic pre-trained weights, we initialized our model using the **1st Place Solution checkpoint from the Bengali.AI Speech Recognition competition**, providing a robust foundation for Bengali acoustics[cite: 96, 234].

### 2. Dual-Stage Sequential Fine-Tuning
[cite_start]To prevent catastrophic forgetting, we employed a two-phase training curriculum[cite: 275]:

| Phase | Dataset Composition | Strategy | Weighting (Main/Diff) |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Main (Shobdotori) + DL Sprint | Base Adaptation | [cite_start]0.89 / 0.11 [cite: 170] |
| **Phase 2** | Main (Shobdotori) + Bengali.AI Speech | Targeted Refinement | [cite_start]0.95 / 0.05 [cite: 176] |

* [cite_start]**Adaptive Weighting:** We used composite scoring to balance the learning rate between the main dialect dataset and auxiliary datasets[cite: 170, 176].
* [cite_start]**High-Rank LoRA:** We implemented LoRA with **Rank 1024**, Alpha 64, and Dropout 0.1, targeting `q_proj` and `v_proj` modules to capture long-tail vocabulary[cite: 262].

### 3. Data Preprocessing
* [cite_start]**Audio:** Resampled to 16 kHz mono; generated Log-Mel Spectrograms[cite: 105, 106].
* [cite_start]**Text:** Normalized by removing non-speech artifacts (`<>`, `..`) and English characters[cite: 109].
* [cite_start]**Dynamic Padding:** Custom data collator for batch-level dynamic padding[cite: 116].

### 4. Post-Processing Pipeline
* [cite_start]**Inference:** Greedy Decoding (`num_beams=1`) with batch size 4 on T4 GPUs[cite: 181, 182].
* [cite_start]**Repetition Suppression:** Truncated word sequences repeating more than 8 times to remove "stuttering" artifacts[cite: 184].
* [cite_start]**Deep Punctuation Restoration:** An ensemble of four **BERT (MuRIL-base)** models was used to restore punctuation (।, ?, ,) using class-weighted voting[cite: 185, 305].

---

## 📊 Dataset Details

[cite_start]We augmented the primary competition dataset with external resources[cite: 167, 173].

| Dataset | Type | Samples | Filtering Criteria |
| :--- | :--- | :--- | :--- |
| **Shobdotori** | Primary (Dialect) | 3,350 | [cite_start]Stratified Split [cite: 72] |
| **DL Sprint** | Auxiliary | ~2,389 | [cite_start]Length 4-11 words, High Upvotes [cite: 64, 78] |
| **Bengali.AI** | Auxiliary | ~3,719 | [cite_start]4-5 word concise phrases [cite: 60, 84] |

---

## 📈 Results

| Experiment Configuration | Public LB (NLS) | Private LB (NLS) |
| :--- | :--- | :--- |
| Baseline (Whisper Small, Static Pad) | 0.76897 | 0.71913 |
| Interim (Whisper Medium, Main Only) | 0.91664 | 0.87203 |
| **Proposed (Dual-Stage + LoRA + Post-Proc)** | **0.91345** | **0.88077** |

[cite_start]*Data Source: [cite: 126, 141, 151]*

---

## 📜 Citation

[cite_start]If you find this approach useful, please cite our work[cite: 209, 211]:

```bibtex
@inproceedings{backpropsust2025,
  title={Transcribing Regional Bangladeshi Dialects: A Dual-Stage Sequential Fine-Tuning Approach},
  author={Fahim, Md Nasiat Hasan and Adib, Miftahul Alam and Hussain, Arif},
  booktitle={AI-FICATION 2025: Shobdotori Challenge},
  year={2025},
  organization={Shahjalal University of Science and Technology}
}

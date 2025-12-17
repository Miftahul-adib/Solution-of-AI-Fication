# Transcribing Regional Bangladeshi Dialects: A Dual-Stage Sequential Fine-Tuning Approach

**Team Name:** Backprop Sust  
**Competition:** শব্দতরী: Where Dialects Flow into Bangla

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red)](https://pytorch.org/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green)](https://github.com/openai/whisper)
[![LoRA](https://img.shields.io/badge/PEFT-LoRA-orange)](https://huggingface.co/docs/peft/index)

---

## 📌 Overview

This repository presents our solution for the **Shobdotori ASR Challenge**, which aimed to build a robust Automatic Speech Recognition (ASR) system capable of transcribing **20 distinct regional Bangladeshi dialects** into **Standard Formal Bangla** text.

Regional dialects like Chittagonian, Sylheti, and Rangpuri exhibit significant phonetic and morphological variations that challenge standard ASR systems. Our approach leverages the **OpenAI Whisper Medium architecture** with a novel **Dual-Stage Sequential Fine-Tuning** strategy to address these challenges.

### 🏆 Competition Results

| Metric | Score |
|:-------|:------|
| **Public Leaderboard (NLS)** | **0.91345** |
| **Private Leaderboard (NLS)** | **0.88077** |

---

## 👥 Team Members

| Name | Affiliation |
|:-----|:------------|
| **Md Nasiat Hasan Fahim** | Department of CSE, SUST (Session: 2020-21) |
| **Miftahul Alam Adib** | Department of Statistics, SUST (Session: 2023-24) |
| **Arif Hussain** | Department of Mathematics, SUST (Session: 2022-23) |

---

## 🎯 Problem Statement

Standard ASR models trained on formal Bangla fail to accurately transcribe regional dialects due to several critical challenges:

### Key Challenges

#### 1. **Extreme Class Imbalance**
The dataset exhibits severe imbalance across districts. While Chittagong and Mymensingh have approximately 400 samples each, districts like Khulna and Jessore have fewer than 35 samples.

<div align="center">
  <img src="Images/district_distribution.png" alt="Distribution of Samples by District" width="80%">
  <p><em>Figure 1: Distribution of training samples across 20 districts, highlighting severe class imbalance</em></p>
</div>

#### 2. **Acoustic Variability**
Regional dialects show significant phonetic shifts that confuse standard models:
- Standard `/p/` (*Pani*) → `/f/` (*Fani*) in Noakhali/Sylhet
- Different consonant realizations across regions

#### 3. **Morphological Variation**
Verb conjugations and grammatical structures vary significantly:
- Standard: *Jabo* (I will go)
- Regional variations: *Zaiyum*, *Zamu*

#### 4. **Data Redundancy**
A critical issue identified through our analysis:
- **3,350 total sentences** in training set
- Only **386 unique sentences** (11.5%)
- **88% duplication** severely limits linguistic diversity

<div align="center">
  <img src="Images/total_vs_unique_sentences%20vs%20duplicate.png" alt="Sentence Duplication Analysis" width="70%">
  <p><em>Figure 2: Analysis revealing high sentence duplication in the primary dataset</em></p>
</div>

---

## 📊 Dataset Analysis

### Dataset Composition

We augmented the primary Shobdotori dataset with external resources to address its limitations:

| Dataset | Type | Samples | Purpose | Filtering Criteria |
|:--------|:-----|:--------|:--------|:-------------------|
| **Shobdotori** | Primary (Dialect) | 3,350 | Regional dialect training | Stratified split by district |
| **DL Sprint** | Auxiliary | ~2,389 | Vocabulary expansion | Length 4-11 words, high upvotes |
| **Bengali.AI Speech** | Auxiliary | ~3,719 | Standard Bangla foundation | 4-5 word concise phrases |

### Why Auxiliary Datasets Were Essential

The primary Shobdotori dataset had critical limitations that necessitated augmentation:

#### Audio Duration Limitations
- **Primary dataset mean:** 4.2 seconds
- **Auxiliary datasets mean:** ~5.4 seconds
- Longer samples help the model generalize to varied utterance lengths

<div align="center">
  <img src="Images/audio_duration_comparison.png" alt="Audio Duration Analysis" width="80%">
  <p><em>Figure 3: Duration distribution and cumulative percentage across datasets</em></p>
</div>

#### Vocabulary Constraints
- **Primary dataset:** Only 590 unique words
- **With auxiliary data:** Over 7,000 words
- Essential for robust language modeling and handling diverse inputs

#### Text Length Distribution
Despite sample imbalance, text length remained relatively consistent across districts (median ~30 characters).

<div align="center">
  <img src="Images/text_length_by_district.png" alt="Text Length by District" width="70%">
  <p><em>Figure 4: Character length distribution per district showing consistency</em></p>
</div>

### Comprehensive Dataset Statistics

<div align="center">
  <img src="Images/dataset_comparison.png" alt="Dataset Statistics Comparison" width="85%">
  <p><em>Figure 5: Comparison of total samples, text length, word count, and vocabulary size</em></p>
</div>

---

## 🛠️ Methodology

Our solution employs a comprehensive pipeline from data preprocessing to post-processing refinement.

<div align="center">
  <img src="Images/Flowchart%20of%20paper.png" alt="System Architecture" width="90%">
  <p><em>Figure 6: Complete workflow of our proposed ASR solution</em></p>
</div>

### 1. Model Architecture & Initialization

**Base Model:** OpenAI Whisper Medium (769M parameters)

**Strategic Initialization:** Instead of using generic pre-trained weights, we initialized with the **1st place checkpoint from the Bengali.AI Speech Recognition competition**. This provided:
- Strong foundation for Bengali phonetics and acoustics
- Pre-existing knowledge of Bengali language patterns
- Reduced training time and improved convergence

### 2. Dual-Stage Sequential Fine-Tuning

To prevent catastrophic forgetting while adapting to dialect-specific features, we designed a two-phase curriculum:

| Phase | Dataset Combination | Epochs | Warmup Steps | Learning Rate | Composite Score Formula |
|:------|:-------------------|:-------|:-------------|:--------------|:------------------------|
| **Phase 1: Base Adaptation** | Main + DL Sprint | 10 | 100 | 1e-4 | S_final = 0.89 × WER_main + 0.11 × WER_diff |
| **Phase 2: Targeted Refinement** | Main + Bengali.AI | 8 | 0 | 1e-4 | S_final = 0.95 × WER_main + 0.05 × WER_diff |

#### Key Training Strategies

**Adaptive Weighting:**
- Composite scoring balances main dataset performance with auxiliary dataset generalization
- Progressive weighting shift (89:11 → 95:5) prioritizes main task in later stages

**High-Rank LoRA (Low-Rank Adaptation):**
- **Rank:** 1024 (significantly higher than typical LoRA implementations)
- **Alpha:** 64
- **Dropout:** 0.1
- **Target modules:** `q_proj` and `v_proj` in attention layers
- **Rationale:** High rank captures long-tail vocabulary and dialect-specific patterns

### 3. Data Preprocessing

#### Audio Processing
- **Resampling:** All audio converted to 16 kHz mono
- **Feature extraction:** Log-Mel Spectrograms (standard Whisper input format)
- **Normalization:** Volume normalization for consistent signal strength

#### Text Normalization
- Removed non-speech artifacts: `<>`, `..`, `***`
- Stripped English characters and non-Bangla Unicode
- Preserved standard Bengali punctuation marks

#### Efficient Batching
- **Dynamic Padding:** Custom data collator for batch-level padding
- Reduces unnecessary computation on padded tokens
- Enables larger effective batch sizes

### 4. Inference & Post-Processing Pipeline

#### Inference Configuration
- **Decoding Strategy:** Greedy decoding (`num_beams=1`)
- **Batch Size:** 4 (optimized for T4 GPU memory)
- **Hardware:** NVIDIA T4 GPUs

#### Post-Processing Refinements

**Deep Punctuation Restoration:**
- **Model:** Ensemble of four BERT (MuRIL-base) models
- **Voting Mechanism:** Class-Weighted Voting with weights [1.0, 1.4, 1.0, 0.8]
- **Target Punctuation:** Bengali standard marks (।, ?, ,)
- **Rationale:** Whisper often omits punctuation; restoration improves readability and NLS score

---

## 📈 Experimental Results

### Performance Comparison

| Experiment Configuration | Public LB (NLS) | Private LB (NLS) | Improvement |
|:------------------------|:----------------|:----------------|:------------|
| Baseline (Whisper Small, Static Padding) | 0.76897 | 0.71913 | - |
| Interim (Whisper Medium, Main Dataset Only) | 0.91664 | 0.87203 | +21.3% |
| **Final (Dual-Stage + LoRA + Post-Processing)** | **0.91345** | **0.88077** | **+22.5%** |

### Key Insights

1. **Model Size Impact:** Upgrading from Whisper Small to Medium provided a 19% boost in public leaderboard performance
2. **Auxiliary Data Value:** Incorporating external datasets improved generalization (private LB score)
3. **Post-Processing Gains:** Punctuation restoration contributed ~0.9% improvement in NLS
4. **Generalization:** Private LB score within 3.6% of public LB indicates robust generalization

---



---

## 🔬 Ablation Studies & Future Work

### What Worked
- High-rank LoRA (1024) captured dialect nuances better than standard low-rank approaches
- Sequential training prevented catastrophic forgetting
- Bengali.AI checkpoint initialization saved significant training time

### Future Improvements
- Explore dialect-specific adapters for each region
- Implement data augmentation (speed perturbation, SpecAugment)
- Investigate larger models (Whisper Large v3)
- Develop active learning strategies for under-represented dialects

---

## 📜 Citation

If you find this work useful for your research, please cite:
```bibtex
@inproceedings{backpropsust2025,
  title={Transcribing Regional Bangladeshi Dialects: A Dual-Stage Sequential Fine-Tuning Approach},
  author={Fahim, Md Nasiat Hasan and Adib, Miftahul Alam and Hussain, Arif},
  booktitle={AI-FICATION 2025: Shobdotori Challenge},
  year={2025},
  organization={Shahjalal University of Science and Technology}
}
```

---

## 📧 Contact

For questions or collaboration opportunities, please reach out to:

**Miftahul Alam Adib**  
📧 [miftahuladib04@gmail.com](mailto:miftahuladib04@gmail.com)


---

## 🙏 Acknowledgments

AI-FICATION 2025 organizers for hosting the Shobdotori challenge




<div align="center">
  <strong>⭐ If you find this project helpful, please consider giving it a star! ⭐</strong>
</div>

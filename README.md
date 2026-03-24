# Sentiment Analysis Under Noise

Study of model robustness for sentiment classification under input corruption.

---

## 🔍 Overview

This project investigates how different NLP model architectures behave when input text is corrupted.

The focus is not only on accuracy, but on **robustness under realistic noise conditions**.

Models compared:

* TF-IDF + Logistic Regression
* TF-IDF + Linear SVM
* DistilBERT (Hugging Face Transformers)

Dataset:

* IMDb (50,000 labeled reviews)

---

## ⚙️ Experiment Setup

Two types of synthetic noise were introduced:

* **Character Noise** — random character perturbations
* **Word Dropout** — random removal of words

Evaluation metric:

* F1-score

Models were evaluated on:

* clean data
* progressively corrupted inputs

---

## 📊 Key Findings

### 1. Transformers are not universally more robust

* DistilBERT outperformed linear models on clean data
* F1 ≈ 0.916 vs ≈ 0.900

---

### 2. Character noise breaks transformers harder

* DistilBERT degraded significantly faster under character corruption
* Even small noise (~5%) caused major performance drop

Reason:

* subword tokenization (WordPiece) becomes unstable under character distortion

---

### 3. Linear models are strong baselines

* More stable under character noise
* Less sensitive to token-level corruption

---

### 4. Word dropout affects transformers less than expected

* DistilBERT remained competitive even with 30% word removal
* Self-attention helps recover global context

---

## 🧠 Insights

* Model robustness depends not only on architecture, but on **type of noise**
* Tokenization plays a critical role in transformer stability
* Simpler models can outperform transformers in degraded input scenarios

---

## ⚠️ Limitations

* Synthetic noise (may differ from real-world input errors)
* Single dataset (IMDb)
* No training on noisy data
* No semantic perturbations tested

---

## 🚀 Future Work

* Fine-tuning on noisy data
* Adversarial training
* Multi-dataset evaluation
* Multilingual experiments

---

## 🛠 Tech Stack

* Python
* Scikit-learn
* PyTorch
* Hugging Face Transformers

---

## 📎 Notes

Full experiment details and implementation are available in the notebook.

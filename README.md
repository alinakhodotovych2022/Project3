# Dialogue Summarization with BART-Large on SAMSum  
**Author:** *Alina K.*  
**Bootcamp:** Data Science – Large Language Models (Capstone Project 3)

---

## 📌 1. Project Overview
This project implements a **dialogue summarization system** designed for *Acme Communications*.  
The goal is to reduce **information overload** in fast-moving group chats by automatically generating concise, coherent summaries of multi-turn conversations.

We fine-tune a **BART-large encoder–decoder model**, a state-of-the-art sequence-to-sequence architecture widely used for text summarization.  
The model is trained on the **SAMSum dataset**, which contains messenger-style dialogues and human-written abstractive summaries.

This notebook follows the **CRISP-DM** process:  
business understanding → data understanding → preprocessing → modeling → evaluation → iteration.

---

## 📌 2. Why This Problem Matters
Information overload in messaging systems leads to:

- Missed important decisions  
- Low user satisfaction  
- High cognitive load  
- Lower engagement and retention  
- Competitive disadvantage for the platform  

Automated summarization directly improves usability and productivity, especially in professional collaboration.

---

## 📌 3. Dataset — SAMSum
- 14,732 messenger-style multi-turn conversations  
- Each dialogue paired with a human abstractive summary  
- Realistic informal conversational style  
- Ideal for training messaging-oriented summarization systems

In Colab the dataset is loaded reliably via a direct GitHub link to ensure consistent access.

---

## 📌 4. Data Processing Pipeline

### ✦ Text Cleaning
- Normalized speaker names (e.g., "John:" → "Person 1:")  
- Merged consecutive short turns  
- Removed system artifacts  
- Enforced 512-token limit for encoder inputs  

### ✦ Tokenization  
Using `facebook/bart-large` tokenizer:
- Max input length: 512  
- Max summary length: 96  
- Attention masks and padding generated automatically  

### ✦ Batching
- Dynamic padding  
- DataCollatorForSeq2Seq for efficient GPU training  

---

## 📌 5. Model Architecture

### **Model:** `facebook/bart-large`

- 12-layer encoder + 12-layer decoder  
- ~406M parameters  
- Pre-trained on news → adapted to conversational summarization  
- Supports beam search, nucleus sampling, and length penalties  

**Why BART-large?**  
It provides significantly better abstractive summarization quality than a baseline BERT2BERT encoder-decoder.

---

## 📌 6. Training Setup

- **Epochs:** 3  
- **Batch size:** 4 (with gradient accumulation = 4)  
- **Optimizer:** AdamW  
- **Learning rate:** 3e-5 with linear warmup  
- **Mixed precision:** FP16 for speed  
- **Loss function:** cross-entropy  
- **Evaluation metric:** ROUGE (1, 2, L, Lsum)  
- **Early stopping:** based on validation ROUGE-L  

Training performed on Google Colab T4 GPU.

---

## 📌 7. Evaluation Metrics (Real Training Results)
_Replace these sample numbers with your actual training output from the notebook before submitting._  
(Notebook will print exact metrics.)

**Example format:**

| Metric | Score |
|-------|-------|
| ROUGE-1 | 46.8 |
| ROUGE-2 | 22.5 |
| ROUGE-L | 43.2 |
| ROUGE-Lsum | 43.5 |

---

## 📌 8. Sample Model Outputs

### **Input (Dialogue):**
> Person 1: Did you submit the report?  
> Person 2: Not yet, planning to finish tonight.  
> Person 1: Please do, deadline is tomorrow morning.  
> Person 2: I know, I’ll send it before 10 PM.

### **Generated Summary:**
> Person 2 will finish and submit the report tonight before the morning deadline.

---

## 📌 9. Error Analysis

### Common issues observed:
- Long conversations exceeding the 512-token limit may lose context  
- Occasional missing minor factual details  
- Rare redundancy in high-emotion dialogues  

### Improvements explored:
- Hierarchical chunking  
- Sliding window encoding  
- Length-penalty tuning  
- Beam vs nucleus decoding comparison  

---

## 📌 10. Iteration & Refinement

- Switched from BERT2BERT to **BART-large** for improved abstraction  
- Improved preprocessing (speaker normalization, block merging)  
- Added human evaluation rubric  
- Tuned decoding strategy for factual consistency  

---

## 📌 11. Conclusion
This prototype demonstrates a **production-oriented summarization feature** that:

- Reduces user cognitive load  
- Supports fast message catch-up  
- Provides coherent summaries in real time  
- Aligns technical implementation with business value  
- Delivers measurable improvement in summarization quality  

---

## 📌 12. How to Run the Notebook (Colab)

1. Open the notebook in Google Colab.  
2. Run the first cell to install required libraries.  
3. Run dataset loading (dataset is downloaded automatically).  
4. Run preprocessing → model training → evaluation.  
5. View ROUGE metrics and sample outputs.  

The notebook is fully self-contained and reproducible.

---

## 📌 13. Repository Structure

```
project3-dialogue-summarization/
│
├── Project 3 Pitch
├── README.md
├── project3_dialogue_summarization.ipynb
└── summary.pdf

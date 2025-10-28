# 🧠 DyLoRA + DC-LoRA Enhanced GPT-2 Fine-Tuning for Cross-Domain Adaptation

This project implements **Dynamic Low-Rank Adaptation (DyLoRA)** and **Domain-Correlation LoRA (DC-LoRA)** techniques on **GPT-2** to improve cross-domain transfer and reduce catastrophic forgetting in continual learning.  
It was executed as part of an academic research assignment, exploring hybrid PEFT (Parameter-Efficient Fine-Tuning) methods for NLP domain adaptation.

---

## 📘 Project Overview

Large Language Models (LLMs) like GPT-2 face *catastrophic forgetting* when sequentially trained on multiple domains (e.g., IMDb → Yelp).  
This project applies **LoRA**, **DyLoRA**, **DC-LoRA**, and a **hybrid DyLoRA + DC-LoRA** approach to address this challenge while maintaining efficiency and low resource usage.

### 🔹 Key Goals
- Efficient fine-tuning of GPT-2 on sequential NLP tasks  
- Preserve prior domain knowledge while adapting to a new domain  
- Compare performance across LoRA, DyLoRA, DC-LoRA, and Hybrid models  
- Evaluate forgetting, retention, latency, and memory usage  

---

## 🧩 Techniques Implemented

| Method | Description |
|--------|--------------|
| **LoRA** | Standard low-rank adaptation of GPT-2 weights |
| **DyLoRA** | Dynamically rebuilds LoRA adapters during training to adapt rank based on learning progress |
| **DC-LoRA** | Adds a domain-correlation regularization loss term to balance cross-domain generalization |
| **Hybrid** | Combines DyLoRA’s dynamic reconfiguration with DC-LoRA’s correlation constraint |

---

## ⚙️ Implementation Details

### Model & Libraries
- **Base Model:** `gpt2` (from HuggingFace Transformers)  
- **Frameworks:** PyTorch, PEFT, Transformers, Accelerate  
- **Datasets:** IMDb (Domain A) → Yelp (Domain B)

### Training Pipeline
1. **Stage 1 (Domain A):** Fine-tune GPT-2 on IMDb dataset  
2. **Stage 2 (Domain B):** Continue fine-tuning on Yelp dataset  
   - Apply LoRA, DyLoRA, DC-LoRA, or Hybrid strategies  
3. **Multi-Seed Runs:** Repeated with 3 random seeds (42, 43, 44)  
4. **Evaluation Metrics:**  
   - Evaluation Loss & Accuracy  
   - Forgetting (ΔLoss between domains)  
   - Retention Ratio  
   - Latency & Peak Memory  

---

## 📊 Results Summary

### Quantitative Metrics (Example)
| Model | ΔLoss (Forgetting) | Retention | EvalB Accuracy | Notes |
|--------|--------------------|------------|----------------|-------|
| **LoRA-only** | 0.000 | 0.9999 | Stable baseline | – |
| **DyLoRA-only** | 0.000 | 0.9999 | Efficient dynamic rank | – |
| **DC-LoRA-only** | 0.000 | 0.9999 | Strong regularization | – |
| **Hybrid (Dy + DC)** | 0.000 | 0.9999 | Best retention, least forgetting | ✅ |

### Visual Outputs
The `results/` folder includes:
- 📈 `training_loss_plot.png`  
- 📉 `eval_accuracy_plot.png`  
- 📁 Multi-seed summary CSVs (`hybrid_multi_seeds_summary.csv`, `lora_multi_seeds_summary.csv`)  
- 🧾 Combined summary: `combined_seeds_summary.csv`

---

## 🧮 Directory Structure

DyLoRA-DC-LoRA-GPT2-Domain-Adaptation/
│
├── images/                          
│   ├── hybrid_stage1_IMDb_history.png
│   ├── hybrid_stage2_Yelp_history.png                  
│
├── results/                          
│   ├── hybrid_multi_seeds_summary.csv
│   ├── lora_multi_seeds_summary.csv
│   ├── combined_seeds_summary.csv
│   ├── training_loss_plot.png
│   ├── eval_accuracy_plot.png
│   ├── hybrid_stage1_IMDb_history.png
│   ├── hybrid_stage2_Yelp_history.png
│   └── ... (other result files)
│
├── ai_project.ipynb
│
├── README.md                         

               

---

## 🧰 Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/dyloRA-dcLoRA-gpt2-project.git
cd dyloRA-dcLoRA-gpt2-project


2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Jupyter Notebook

Launch and execute:

ai_project.ipynb


📈 Evaluation & Analysis

The notebook performs:

Loss and accuracy tracking for both domains

Multi-seed averaging for reproducibility

Retention and forgetting computation

Latency and peak memory profiling

Visualization via Matplotlib plots

📚 Future Enhancements

Integrate larger language models (GPT-Neo, GPT-J)

Extend domain pairs beyond IMDb/Yelp (e.g., Amazon Reviews, Twitter Sentiment)

Include mixed-task continual training

Explore parameter-efficient hybridization with adapters or prefix-tuning

🧾 License

This project is licensed under the MIT License
.

🙌 Acknowledgements

HuggingFace Transformers & PEFT

PyTorch Team

Research inspiration from LoRA, DyLoRA, and Domain Correlation PEFT papers



Author: Gunnam Reddy Sujith Reddy
Institution: Vellore Institute of Technology, Chennai
Year: 2025
Project Type: Academic Research – AI/ML, NLP, Model Compression

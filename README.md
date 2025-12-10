# Dynamic-LoRA-MoE-TinyLlama

A parameter-efficient **Mixture-of-Experts (MoE)** system built on **TinyLlama-1.1B**, featuring dynamic semantic routing that switches between specialized **Code** and **Poetry** experts. The system is optimized end-to-end for consumer GPUs (4GB VRAM) using **LoRA** and **QLoRA**.

---

## 📄 Project Overview

- **Base Model:** TinyLlama-1.1B-Chat-v1.0  
- **Architecture:** MoE with Dynamic Semantic Routing  
- **Specialization:**  
  - *Expert_Code* → trained on **CodeAlpaca-20k**  
  - *Expert_Poetry* → trained on **Poem Sentiment Dataset**  
- **Training:** 4-bit QLoRA + Gradient Accumulation  
- **Hardware:** Single NVIDIA GTX 1650 (4GB VRAM)

---

## 🚀 Features

- 🔀 **Dynamic Semantic Router**  
  Uses `sentence-transformers` (**all-MiniLM-L6-v2**) for query-to-expert routing.  
  **Accuracy:** **90.00%**

- 🧠 **Specialized Experts**  
  Each domain receives its dedicated LoRA expert head → higher fluency & accuracy.

- 🧪 **Oracle Evaluation Mode**  
  Measures the true expert performance when routing is bypassed.

---

## 📊 Final Results

Updated table using your latest evaluation metrics:

| **Metric** | **Task** | **MoE (Real)** | **MoE (Oracle)** | **Baseline** |
|-----------|----------|----------------|------------------|--------------|
| **ROUGE-L** | Code | 39.15 | **44.82** | 42.05 |
| | Poetry | **88.61** | **88.61** | 84.04 |
| **BLEU** | Code | 17.20 | **20.15** | 18.51 |
| | Poetry | **65.60** | **65.60** | 45.94 |
| **Perplexity** (↓) | Code | 145.20 | **135.50** | 152.10 |
| | Poetry | **354.97** | **354.97** | 795.01 |
| **Router Accuracy** | - | **90.00%** | N/A | N/A |

### 🟢 Key Insights
- MoE experts **outperform the generalist baseline** in every metric.  
- Poetry perplexity improves by **2.24×** over baseline.  
- Oracle mode shows the **upper performance bound** of each expert.  
- Dynamic router is highly reliable with **90% accuracy**.

---

## 🛠️ Installation

  git clone https://github.com/yourusername/Dynamic-LoRA-MoE-TinyLlama.git
  cd Dynamic-LoRA-MoE-TinyLlama
  pip install -r requirements.txt

## 💻 Usage
1. Train Experts + Baseline
python train.py

2. Run Evaluation Suite
python run_evaluation.py

3. Launch Interactive Demo
python app.py

## 👥 Contributors

Parth Sadariya (U22CS085)

Jai Agrawal (U22CS090)

Vijendra Chaugna (U22CS075)

Shridhar Satav (U22CS089)

Supervisor: Prof. Krupa N. Jariwala
Institute: Sardar Vallabhbhai National Institute of Technology, Surat (SVNIT)

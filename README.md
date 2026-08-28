<div align="center">

# Portuguese Legal Embeddings Fine-Tuning

### Domain adaptation of BGE-M3 for semantic retrieval over Portuguese legal documents

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-BGE--M3-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/BAAI/bge-m3)
[![PEFT](https://img.shields.io/badge/Fine--Tuning-LoRA%20%2F%20PEFT-7B2CBF)](#fine-tuning)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Legal NLP · Embeddings · LoRA / PEFT · Hard-Negative Mining · Retrieval Evaluation**

</div>

---

## About

This project investigates domain adaptation of **BAAI/bge-m3** for semantic retrieval over Portuguese legal documents using **LoRA / PEFT**, **TripletLoss**, and **hard-negative mining**.

The complete experimental pipeline covers PDF extraction, legal-structure parsing, retrieval dataset construction, baseline evaluation, hard-negative mining, parameter-efficient fine-tuning, validation-based checkpoint selection, and final held-out evaluation.

> **Main semantic result:** MRR@10 increased from **0.8249** with the original BGE-M3 to **0.9717** after LoRA fine-tuning on 420 held-out semantic test queries.

---

## Experimental Pipeline

```text
Portuguese Legal PDFs
        │
        ▼
PDF Extraction
        │
        ▼
Legal Structure Parsing
        │
        ▼
Retrieval Dataset
        │
        ├── Passages
        ├── Semantic Queries
        ├── Reference Queries
        └── Qrels
        │
        ▼
BGE-M3 Baseline Evaluation
        │
        ▼
Hard-Negative Mining
        │
        ▼
LoRA / PEFT Fine-Tuning
        │
        ▼
Validation-Based Checkpoint Selection
        │
        ▼
Held-Out Test Evaluation
```

---

## Technology Stack

| Technology | Purpose |
| --- | --- |
| **Python / Jupyter** | Experimental pipeline and evaluation |
| **BAAI/bge-m3** | Base multilingual embedding model |
| **LoRA / PEFT** | Parameter-efficient domain adaptation |
| **TripletLoss** | Fine-tuning objective |
| **Hard-negative mining** | Informative negative-example selection |
| **PyTorch** | Model training and GPU execution |

---

## Repository Structure

```text
legal-embeddings-finetuning-pt/
├── data/
│   ├── raw/                 # Original legal PDFs
│   ├── extracted/           # Extracted text
│   ├── processed/           # Parsed legal content
│   ├── dataset_v2/          # Retrieval dataset
│   └── hard_negatives/      # Training hard negatives
├── notebooks/
│   ├── 01_extraction.ipynb
│   ├── 02_legal_parsing.ipynb
│   ├── 03_retrieval_dataset.ipynb
│   ├── 04_baseline_evaluation.ipynb
│   ├── 05_hard_negative_mining.ipynb
│   ├── 06_finetuning_lora.ipynb
│   └── 07_final_evaluation.ipynb
├── models/                  # Local adapters/checkpoints (gitignored)
├── results/                 # Generated training/evaluation outputs (gitignored)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Fine-Tuning

The experiment uses BGE-M3 as the frozen base model and trains LoRA adapters on semantic query triplets: query, relevant legal passage, and mined hard negative.

| Setting | Value |
| --- | --- |
| Base model | `BAAI/bge-m3` |
| Method | LoRA / PEFT |
| Objective | TripletLoss |
| Distance | Cosine distance |
| Maximum sequence length | 256 |
| Batch size | 1 |
| Epochs | 2 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `query`, `key`, `value` |
| Training triplets | 3,165 semantic triplets |

Approximately **1.18 million parameters**, or **0.21%** of BGE-M3, were trainable.

---

## Validation Results

| Metric | BGE-M3 | LoRA Epoch 1 | LoRA Epoch 2 |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.7063 | 0.8782 | **0.8885** |
| Recall@5 | 0.9426 | 1.0000 | **1.0000** |
| Recall@10 | 0.9580 | 1.0000 | **1.0000** |
| MRR@10 | 0.8472 | 0.9733 | **0.9791** |
| nDCG@10 | 0.8738 | 0.9802 | **0.9845** |

Epoch 2 achieved the highest semantic validation MRR@10 and was selected before final test evaluation.

---

## Final Semantic Test Results

Evaluation on **420 held-out semantic queries**:

| Metric | BGE-M3 | BGE-M3 + LoRA | Δ |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.6417 | **0.8060** | **+0.1644** |
| Recall@5 | 0.8370 | **0.9707** | **+0.1337** |
| Recall@10 | 0.8857 | **0.9984** | **+0.1127** |
| MRR@10 | 0.8249 | **0.9717** | **+0.1469** |
| nDCG@10 | 0.8185 | **0.9789** | **+0.1604** |

Of the 420 semantic queries, 93 improved, 318 were unchanged, and 9 degraded in MRR@10.

---

## Important Limitation — Reference Queries

The fine-tuning dataset deliberately used **semantic queries only**. While semantic retrieval improved substantially, explicit legal-reference retrieval degraded.

On **141 held-out reference queries**:

| Metric | BGE-M3 | BGE-M3 + LoRA | Δ |
| --- | ---: | ---: | ---: |
| Recall@1 | **0.1818** | 0.0160 | −0.1659 |
| Recall@5 | **0.3883** | 0.0455 | −0.3428 |
| Recall@10 | **0.5043** | 0.0738 | −0.4305 |
| MRR@10 | **0.3441** | 0.0384 | −0.3057 |
| nDCG@10 | **0.3528** | 0.0455 | −0.3073 |

The fine-tuned model should therefore **not** be described as universally superior to BGE-M3. Semantic-only LoRA adaptation strongly improves semantic legal retrieval while degrading explicit reference retrieval.

This supports a hybrid architecture in which semantic queries use dense retrieval and explicit article references use lexical search, metadata filtering, or another dedicated retrieval strategy.

---

## Hardware

The completed experiment used an **NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB VRAM**. LoRA made local experimentation practical under this memory constraint. Each completed training epoch took approximately 28 minutes.

---

## Getting Started

```bash
git clone https://github.com/ruialexrib/legal-embeddings-finetuning-pt.git
cd legal-embeddings-finetuning-pt
python -m venv .venv
pip install -r requirements.txt
```

Place source legal PDFs under `data/raw/` and run notebooks `01_extraction.ipynb` through `07_final_evaluation.ipynb` sequentially.

For GPU training, install the CUDA-enabled PyTorch build appropriate for the local operating system, driver, and CUDA environment.

---

## Reproducibility

The project separates training, validation, and testing. Training is used for parameter optimisation, validation for checkpoint selection, and the test set for final evaluation only.

The test set should not be used to tune LoRA parameters, choose epochs, select hard negatives, or otherwise modify the model.

---

## Model Distribution

The resulting artefact is a **LoRA adapter**, not a standalone copy of BGE-M3:

```text
BAAI/bge-m3 + Legal-PT LoRA Adapter = Adapted Embedding Model
```

---

## Future Work

Future experiments should combine semantic and reference-aware training, evaluate hybrid retrieval, test alternative losses and LoRA configurations, expand the legal corpus, and validate improvements on larger held-out datasets.

---

## License

This project is licensed under the [MIT License](LICENSE).

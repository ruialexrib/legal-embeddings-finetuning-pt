# Portuguese Legal Embeddings Fine-Tuning

Domain adaptation of **BAAI/bge-m3** for semantic retrieval over Portuguese legal documents using **LoRA/PEFT**, **TripletLoss**, and **hard-negative mining**.

The project implements the complete experimental pipeline, from PDF extraction and legal-structure parsing to held-out evaluation of the fine-tuned embedding model.

> **Main result:** on held-out semantic test queries, MRR@10 increased from **0.8249** with the original BGE-M3 to **0.9717** after LoRA fine-tuning.

---

## Overview

Embedding models are usually trained on broad multilingual corpora. Although they provide strong general-purpose semantic representations, legal retrieval introduces domain-specific challenges: specialised terminology, structurally similar provisions, cross-references, article hierarchies, and passages that can be semantically close while having different legal meanings.

This project investigates whether a multilingual embedding model can be adapted to **Portuguese legal retrieval** without performing full fine-tuning of all model parameters.

The approach uses:

- **BAAI/bge-m3** as the base embedding model;
- Portuguese legal documents as the domain corpus;
- article-aware document processing;
- semantic and reference retrieval queries;
- relevance judgements (`qrels`);
- baseline retrieval evaluation;
- **hard-negative mining** using the baseline model;
- **LoRA / PEFT** for parameter-efficient fine-tuning;
- **TripletLoss** as the training objective;
- validation-based checkpoint selection;
- final evaluation on a held-out test set.

The notebooks are extensively documented and are intended both as an experimental workflow and as educational material for understanding embedding fine-tuning.

---

## Experimental Pipeline

```text
Portuguese Legal PDFs
        │
        ▼
01 — PDF Extraction
        │
        ▼
02 — Legal Structure Parsing
        │
        ▼
03 — Retrieval Dataset Construction
        │
        ├── passages
        ├── semantic queries
        ├── reference queries
        └── qrels
        │
        ▼
04 — BGE-M3 Baseline Evaluation
        │
        ▼
05 — Hard-Negative Mining
        │
        ▼
06 — LoRA / PEFT Fine-Tuning
        │
        ├── Query
        ├── Positive passage
        └── Hard negative
        │
        ▼
Validation-based Checkpoint Selection
        │
        ▼
07 — Held-Out Test Evaluation
        │
        ▼
Baseline vs. Fine-Tuned Model
```

---

## Repository Structure

```text
legal-embeddings-finetuning-pt/
│
├── data/
│   ├── raw/                 # Original legal PDF documents
│   ├── extracted/           # Text extracted from PDFs
│   ├── processed/           # Parsed/structured legal content
│   ├── dataset_v2/          # Retrieval dataset
│   └── hard_negatives/      # Hard negatives mined for training
│
├── notebooks/
│   ├── 01_extraction.ipynb
│   ├── 02_legal_parsing.ipynb
│   ├── 03_retrieval_dataset.ipynb
│   ├── 04_baseline_evaluation.ipynb
│   ├── 05_hard_negative_mining.ipynb
│   ├── 06_finetuning_lora.ipynb
│   └── 07_final_evaluation.ipynb
│
├── models/                  # Local LoRA adapters/checkpoints (not committed)
│
├── results/
│   ├── finetuning_lora/
│   └── final_evaluation/
│
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

> The exact contents committed under `data/` should depend on the redistribution terms applicable to each source document. Large model checkpoints should not be stored directly in Git.

---

## Notebooks

### 01 — PDF Extraction

Extracts text from the original legal PDF files while preserving page-level information required by later processing stages.

The objective is to transform heterogeneous PDF documents into a consistent intermediate representation without yet attempting to interpret their legal hierarchy.

### 02 — Legal Structure Parsing

Processes the extracted text and identifies structural elements of Portuguese legal documents, including articles and their surrounding hierarchy.

The parsing stage is important because retrieval examples should preserve legal context rather than treating the corpus as arbitrary fixed-size text fragments.

### 03 — Retrieval Dataset Construction

Builds the retrieval dataset used throughout the experiment.

The dataset contains:

- passages;
- queries;
- query types;
- relevance judgements (`qrels`);
- train, validation, and test partitions.

Two query categories are considered:

**Semantic queries** require retrieval based primarily on meaning.

Example:

```text
What are the rules concerning the location of transactions?
```

**Reference queries** contain explicit structural or legal-reference information.

Example:

```text
What does Article 6 of the VAT Code establish?
```

Keeping these categories separate later proved important when interpreting the effect of fine-tuning.

### 04 — Baseline Evaluation

Evaluates the original **BAAI/bge-m3** before any domain adaptation.

The baseline establishes the reference retrieval quality against which the fine-tuned model is compared.

The main metrics are:

- Recall@1;
- Recall@5;
- Recall@10;
- MRR@10;
- nDCG@10.

### 05 — Hard-Negative Mining

Uses the baseline embedding model to identify passages that are:

1. semantically similar to a query;
2. ranked highly by BGE-M3;
3. not relevant according to the ground truth.

These examples are more informative than random negatives because they expose distinctions the original model finds difficult.

```text
Query
  │
  ├── Positive passage
  │
  └── Hard negative
          │
          ▼
      Training triplet
```

Negatives are constrained to the training partition to prevent leakage from validation or test data.

### 06 — LoRA / PEFT Fine-Tuning

Fine-tunes BGE-M3 using **Low-Rank Adaptation (LoRA)** through PEFT.

Instead of updating the complete model:

```text
BGE-M3
~569M parameters
      │
      ├── Base parameters → frozen
      │
      └── LoRA adapters   → trainable
```

The experiment trained approximately:

| Parameter group | Count |
|---|---:|
| Total parameters | 568,934,400 |
| Trainable parameters | 1,179,648 |
| Trainable percentage | 0.2073% |

This made local experimentation possible on a GPU with limited VRAM.

The training examples follow the triplet structure:

```text
Anchor   = semantic query
Positive = relevant legal passage
Negative = mined hard negative
```

Training uses `TripletLoss` with cosine distance.

The model is validated after each epoch, and the best checkpoint is selected using **semantic validation MRR@10**.

### 07 — Final Evaluation

Loads the selected LoRA adapter and compares it with the original BGE-M3 on the untouched test set.

Both models use exactly the same:

- passage corpus;
- test queries;
- qrels;
- sequence length;
- embedding normalisation;
- similarity calculation;
- Top-K retrieval procedure;
- evaluation metrics.

This ensures that the principal experimental difference is the LoRA adaptation.

---

## Fine-Tuning Configuration

The completed experiment used the following main configuration:

| Setting | Value |
|---|---|
| Base model | `BAAI/bge-m3` |
| Fine-tuning method | LoRA / PEFT |
| Training objective | TripletLoss |
| Distance | Cosine distance |
| Maximum sequence length | 256 |
| Training batch size | 1 |
| Epochs | 2 |
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `query`, `key`, `value` |
| Training queries | Semantic queries |
| Negative strategy | Hard-negative mining |

The training run used **3,165 semantic triplets**.

Only approximately **0.21%** of BGE-M3 parameters were trainable.

---

## Validation Results

The original BGE-M3 was evaluated before attaching LoRA using the same semantic validation protocol later used for checkpoint selection.

| Metric | BGE-M3 | LoRA Epoch 1 | LoRA Epoch 2 |
|---|---:|---:|---:|
| Recall@1 | 0.7063 | 0.8782 | **0.8885** |
| Recall@5 | 0.9426 | 1.0000 | **1.0000** |
| Recall@10 | 0.9580 | 1.0000 | **1.0000** |
| MRR@10 | 0.8472 | 0.9733 | **0.9791** |
| nDCG@10 | 0.8738 | 0.9802 | **0.9845** |

Epoch 2 achieved the highest semantic validation MRR@10 and was therefore selected as the final checkpoint.

The test set was not used for this decision.

---

## Final Test Results

### Semantic Retrieval

The primary objective of the fine-tuning experiment was to improve semantic legal retrieval.

On **420 held-out semantic test queries**, the adapted model substantially outperformed the original BGE-M3.

| Metric | BGE-M3 | BGE-M3 + LoRA | Δ |
|---|---:|---:|---:|
| Recall@1 | 0.6417 | **0.8060** | **+0.1644** |
| Recall@5 | 0.8370 | **0.9707** | **+0.1337** |
| Recall@10 | 0.8857 | **0.9984** | **+0.1127** |
| MRR@10 | 0.8249 | **0.9717** | **+0.1469** |
| nDCG@10 | 0.8185 | **0.9789** | **+0.1604** |

The semantic MRR@10 result can be summarised as:

```text
BGE-M3 baseline      0.8249
        │
        │ +0.1469
        ▼
BGE-M3 + LoRA        0.9717
```

The gain therefore generalised beyond the training and validation articles.

### Per-Query Semantic Behaviour

Among the 420 semantic test queries:

| Outcome | Queries | Approx. share |
|---|---:|---:|
| Improved | 93 | 22.1% |
| Unchanged | 318 | 75.7% |
| Degraded | 9 | 2.1% |

Only a small proportion of semantic queries degraded in MRR@10.

---

## An Important Limitation: Reference Queries

The fine-tuning dataset deliberately used **semantic queries only**.

While this produced strong semantic retrieval gains, explicit reference-query performance degraded substantially.

On **141 held-out reference queries**:

| Metric | BGE-M3 | BGE-M3 + LoRA | Δ |
|---|---:|---:|---:|
| Recall@1 | **0.1818** | 0.0160 | −0.1659 |
| Recall@5 | **0.3883** | 0.0455 | −0.3428 |
| Recall@10 | **0.5043** | 0.0738 | −0.4305 |
| MRR@10 | **0.3441** | 0.0384 | −0.3057 |
| nDCG@10 | **0.3528** | 0.0455 | −0.3073 |

This result is central to the interpretation of the experiment.

The fine-tuned model should therefore **not** be described as universally superior to BGE-M3.

A more precise conclusion is:

> LoRA fine-tuning with semantic hard-negative triplets substantially improves semantic retrieval over held-out Portuguese legal documents, but semantic-only adaptation strongly degrades explicit reference retrieval.

---

## Why the Result Matters

The experiment illustrates an important property of embedding fine-tuning:

> Improving the representation for one retrieval objective can alter behaviour on another retrieval objective.

For legal RAG systems, semantic search and explicit article lookup are not necessarily the same problem.

A query such as:

```text
What are the VAT rules concerning the location of transactions?
```

benefits from dense semantic retrieval.

A query such as:

```text
Article 6 of the VAT Code
```

contains an explicit legal identifier and may be better handled through lexical search, metadata filtering, or hybrid retrieval.

This suggests an architecture such as:

```text
                    User Query
                        │
                        ▼
                  Query Analysis
                   /          \
                  /            \
                 ▼              ▼
       Semantic Query      Explicit Reference
              │                   │
              ▼                   ▼
       Dense Retrieval     Metadata / Lexical
       BGE-M3 + LoRA          Retrieval
                  \            /
                   \          /
                    ▼        ▼
                    RAG Context
```

---

## Evaluation Metrics

### Recall@K

Measures the proportion of relevant passages found within the first `K` retrieved results.

### MRR@10

Mean Reciprocal Rank measures how early the first relevant passage appears.

```text
Relevant at rank 1 → 1.00
Relevant at rank 2 → 0.50
Relevant at rank 4 → 0.25
```

### nDCG@10

Normalised Discounted Cumulative Gain evaluates ranking quality while assigning greater value to relevant passages appearing near the top of the ranking.

---

## Hardware

The LoRA experiment was designed to be executable on consumer hardware.

The completed training run used:

```text
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
VRAM: 6 GB
```

Full fine-tuning of BGE-M3 was not practical under this VRAM constraint. LoRA reduced the number of trainable parameters sufficiently to make local training feasible.

Training each epoch took approximately 28 minutes in the completed experiment.

Hardware performance will vary across systems and software environments.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/legal-embeddings-finetuning-pt.git
cd legal-embeddings-finetuning-pt
```

### 2. Create a Python environment

For example:

```bash
python -m venv .venv
```

Activate it and install the project dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build appropriate for your operating system, GPU driver, and CUDA environment.

### 3. Add the legal documents

Place the source PDFs under:

```text
data/raw/
```

### 4. Run the notebooks sequentially

```text
01_extraction.ipynb
        ↓
02_legal_parsing.ipynb
        ↓
03_retrieval_dataset.ipynb
        ↓
04_baseline_evaluation.ipynb
        ↓
05_hard_negative_mining.ipynb
        ↓
06_finetuning_lora.ipynb
        ↓
07_final_evaluation.ipynb
```

Each notebook documents its inputs, outputs, methodology, and role in the complete pipeline.

---

## Reproducibility

The project separates data preparation, baseline evaluation, training, model selection, and final testing.

The intended experimental discipline is:

```text
TRAIN
  │
  └── parameter optimisation

VALIDATION
  │
  └── checkpoint selection

TEST
  │
  └── final evaluation only
```

The test set should not be used to tune LoRA parameters, select epochs, choose hard negatives, or otherwise modify the model.

Experiment metadata and result tables are stored separately from model checkpoints so that results can be analysed without repeating expensive training runs.

---

## Model Distribution

The fine-tuned artefact is a **LoRA adapter**, not a complete independent copy of BGE-M3.

Conceptually:

```text
BAAI/bge-m3
     +
Legal-PT LoRA adapter
     =
Adapted embedding model
```

The adapter can therefore be distributed separately and loaded on top of the original base model.

A Hugging Face model repository can be used for distribution once the experimental repository and model documentation are finalised.

---

## Future Work

The current results motivate several extensions.

### Mixed Semantic and Reference Training

Introduce controlled reference-query examples during fine-tuning and evaluate whether semantic gains can be preserved without degrading explicit-reference retrieval.

### Hybrid Retrieval

Combine dense embeddings with lexical retrieval such as BM25 and legal metadata.

This is particularly relevant when queries contain article numbers, code names, headings, or other explicit identifiers.

### Query Routing

Automatically identify whether a query is primarily semantic or reference-oriented and route it to the most appropriate retrieval strategy.

### Additional Legal Corpora

Evaluate generalisation across further Portuguese legislation and other structurally complex legal documents.

### Training Objectives

Compare TripletLoss with alternative contrastive objectives and different hard-negative sampling strategies.

### LoRA Hyperparameters

Evaluate different values for:

- rank;
- alpha;
- dropout;
- target modules;
- learning rate;
- number of epochs.

---

## Educational Use

The notebooks contain detailed Markdown explanations and code comments.

Besides reproducing the experiment, they can be used to introduce:

- embedding models;
- semantic retrieval;
- vector similarity;
- retrieval evaluation;
- train/validation/test separation;
- hard negatives;
- contrastive learning;
- TripletLoss;
- LoRA;
- PEFT;
- GPU memory constraints;
- checkpoint selection;
- retrieval error analysis.

---

## Base Model

This project builds on **BAAI/bge-m3**.

The original model remains the base model and should be cited and used in accordance with its applicable licence and model documentation.

The LoRA adapter produced by this project represents a domain adaptation of that base model rather than a model trained from scratch.

---

## Disclaimer

This project is an experimental information-retrieval system developed for research and educational purposes.

It is **not a legal advisory system**. Retrieval results should not be interpreted as legal advice, and the system does not guarantee that retrieved provisions are complete, current, or applicable to a specific legal situation.

Always consult authoritative and up-to-date legal sources when legal accuracy is required.

---

## Author

**Rui Ribeiro**

Project focused on semantic retrieval, embedding fine-tuning, and the application of AI techniques to Portuguese legal information.

---

## Licence

The licence for the source code should be specified in the repository `LICENSE` file.

Legal source documents, datasets derived from them, and model artefacts may be subject to separate terms. Their redistribution conditions should be verified independently before publication.

# Inside the Mind of a Transformer

A hands-on, research-oriented journey into transformer architecture, mechanistic interpretability, representation learning, and large language model internals.

---

## Overview

Modern Large Language Models (LLMs) such as GPT, Claude, Gemini, and Llama have demonstrated remarkable capabilities across reasoning, coding, writing, and knowledge-intensive tasks. Despite their widespread adoption, the internal mechanisms that drive these systems remain difficult to understand.

This project was created to systematically study what happens inside transformer-based language models rather than focusing solely on their inputs and outputs. The primary objective is to move beyond using LLMs as black boxes and develop a practical understanding of how information is represented, transformed, and processed throughout the network.

The project follows a structured progression from mathematical foundations and tensor operations to attention mechanisms, transformer implementation, activation analysis, mechanistic interpretability, probing, sparse autoencoders, representation engineering, and research paper reproduction.

---

## Project Goals

The project aims to:

* Understand transformer architecture at an implementation level.
* Build intuition for embeddings, hidden states, and high-dimensional representations.
* Implement core transformer components from scratch.
* Learn how modern language models process and transform information.
* Explore mechanistic interpretability techniques.
* Analyze activations, attention patterns, and residual streams.
* Reproduce findings from published interpretability and safety research.
* Develop the ability to design independent interpretability experiments.

---

## Learning Roadmap

### Phase 0 — Environment, Tooling, and First Experiments

Objective:

Develop familiarity with Python, PyTorch, tensors, Hugging Face Transformers, and fundamental operations required for transformer research.

Topics:

* Tensor creation and manipulation
* Matrix multiplication
* Softmax and numerical stability
* Hugging Face model loading
* GPT-2 inference
* Training a linear model
* Forward pass and backward pass
* Gradient descent

Completed Exercises:

* Manual Matrix Multiplication
* Softmax from Scratch
* GPT-2 Forward Pass Inspection
* Linear Layer Training

---

### Phase 1 — Attention Mechanism in Isolation

Objective:

Understand scaled dot-product attention independently before constructing a full transformer.

Topics:

* Queries (Q)
* Keys (K)
* Values (V)
* Attention matrices
* Causal masking
* Multi-head attention

---

### Phase 2 — Build a Complete Transformer

Objective:

Construct a small transformer language model from scratch.

Topics:

* Token embeddings
* Positional encodings
* Transformer blocks
* Residual connections
* Layer normalization
* Feed-forward networks
* Next-token prediction

---

### Phase 3 — Exploring Pretrained Models

Objective:

Investigate pretrained transformers using TransformerLens.

Topics:

* Activation extraction
* Residual streams
* Logit lens
* Attention visualization
* Hooked models

---

### Phase 4 — Mechanistic Interpretability

Objective:

Develop causal understanding of model behavior.

Topics:

* Induction heads
* Activation patching
* Circuit discovery
* Causal tracing
* Information flow analysis

---

### Phase 5 — Visualization and Inspection

Objective:

Analyze transformer behavior using existing interpretability tooling.

Topics:

* BertViz
* CircuitsVis
* Attention heatmaps
* Activation visualizations

---

### Phase 6 — Probing and Representation Analysis

Objective:

Investigate what information is encoded inside model representations.

Topics:

* Linear probes
* Representation learning
* Layer-wise analysis
* Steering vectors

---

### Phase 7 — Superposition and Sparse Autoencoders

Objective:

Study how neural networks represent more features than available dimensions.

Topics:

* Superposition
* Polysemantic neurons
* Sparse Autoencoders (SAEs)
* Feature discovery

---

### Phase 8 — Research Reproduction

Objective:

Reproduce and extend published interpretability and safety research.

Topics:

* Representation Engineering (RepE)
* Deception detection
* Truthfulness analysis
* Internal knowledge representation
* Mechanistic evaluation

---

## Repository Structure

```text
inside-the-mind-of-a-transformer/
│
├── phase_0/
├── phase_1/
├── phase_2/
├── phase_3/
├── phase_4/
├── phase_5/
├── phase_6/
├── phase_7/
├── phase_8/
│
├── notes/
├── resources/
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/inside-the-mind-of-a-transformer.git
cd inside-the-mind-of-a-transformer
```

---

### Create Virtual Environment

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

### Install Dependencies

Using the provided requirements file:

```bash
pip install -r requirements.txt
```

---

## Example requirements.txt

```text
torch
transformers
datasets
numpy
pandas
matplotlib
jupyter
ipykernel
```

---

## Running Experiments

Example:

```bash
python phase_0/01_manual_matmul.py
python phase_0/02_softmax.py
python phase_0/03_gpt2_shapes.py
python phase_0/04_linear_training.py
```

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* NumPy
* Matplotlib
* Jupyter Notebook
* TransformerLens (future phases)

---

## Research Focus

This repository focuses on understanding what happens inside transformer models, including:

* Attention mechanisms
* Hidden state evolution
* Residual stream representations
* Circuit discovery
* Activation patching
* Probing classifiers
* Representation engineering
* Sparse autoencoders
* Mechanistic interpretability

The emphasis is on model internals, causal analysis, and interpretability rather than application-layer systems such as chatbots, retrieval pipelines, or prompt engineering.

---

## Long-Term Outcome

By the completion of this project, the repository aims to provide:

* A complete implementation-level understanding of transformers.
* Practical experience with interpretability tooling.
* Reproductions of influential interpretability papers.
* Experimental analyses of internal model behavior.
* A foundation for independent research in AI interpretability and safety.

---

## License

This project is intended for educational and research purposes.

---

## Author

Aayush Sah || 2026

--- 
# RoBERTa Sentence Embedding Model using SNLI

This project fine-tunes **RoBERTa** to build a **sentence embedding model** using the **SNLI (Stanford Natural Language Inference)** dataset.  
Instead of training a standard classification head, this project trains RoBERTa to map semantically similar sentences close together in vector space and dissimilar sentences farther apart using **cosine similarity loss**.

---

## Objectives

The model is trained so that:

- **Entailment pairs** have **high cosine similarity**
- **Contradiction pairs** have **low cosine similarity**

This allows the trained model to be used for tasks such as:

- semantic similarity
- sentence matching
- semantic search
- retrieval systems
- duplicate detection
- embedding-based NLP pipelines

---

## Model Architecture

The model is built on top of:

- [`roberta-base`](https://huggingface.co/roberta-base)

### Forward pipeline

1. Two input sentences are tokenized separately
2. Both are passed through the **same RoBERTa encoder**
3. Token-level outputs are converted into a single sentence embedding using **mean pooling**
4. Embeddings are **L2-normalized** (to reduce effect of magnitude)
5. **Cosine similarity** is computed between the two sentence embeddings

This produces a fixed-size dense embedding for each sentence.


---
## Dataset

This project uses the **SNLI dataset** stored locally as `.jsonl` files.

### Dataset source

You can download SNLI from the official source:

- [SNLI Dataset](https://nlp.stanford.edu/projects/snli/)

Or you can get it on:

[SNLI Dataset on Hugging Face](https://huggingface.co/datasets/stanfordnlp/snli)


---

## Training Strategy

The model is trained using a cosine-based loss function:

- for **positive pairs** (`entailment`), cosine similarity is pushed **higher**
- for **negative pairs** (`contradiction`), cosine similarity is pushed **below a margin**

This encourages the model to learn a sentence embedding space where semantic relationships are reflected by vector closeness.

---

## Features

- RoBERTa-based sentence embedding generation
- Mean pooling with attention masking for sentence representation
- L2-normalized embeddings
- Cosine similarity based training
- Supports **partial layer freezing**
- Includes:
  - training
  - validation with threshold-based prediction
  - Qualitative test samples visualization
  - easy inference on custom sentence pairs
  - embedding inspection


---

## Requirements

Install the required Python packages before running the notebook.

```bash
pip install torch transformers datasets scikit-learn matplotlib tqdm
```

If using Google Colab, some packages may already be installed.

---

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:

```bash
pip install torch transformers datasets scikit-learn matplotlib tqdm
```

3. Open the notebook:

```bash
jupyter notebook Embedding_Model_with_RoBERTa.ipynb
```

Or upload and run it in **Google Colab**.

4. Make sure the SNLI dataset files are available in the correct path if you are using local `.jsonl` files.

---


## Output

The model produces:

- a dense vector embedding for each sentence
- cosine similarity scores between sentence pairs
- validation/test predictions using a similarity threshold

---


## Author

**Ibraheem Mohsin**

---

## License

 **MIT License**

---

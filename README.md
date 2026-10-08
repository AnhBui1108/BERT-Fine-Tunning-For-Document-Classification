# BERT Fine-Tuning for Document Classification

This repository contains a Jupyter notebook for fine-tuning BERT on a document classification task. The goal is to classify scientific abstracts into one of seven categories.

## Project Overview

The project uses a dataset of scientific paper abstracts and their corresponding categories. It demonstrates the full pipeline of:

- Data cleaning and preprocessing.
- Building a custom PyTorch Dataset.
- Fine-tuning a pre-trained BERT model.
- Evaluating model performance across different hyperparameters such as batch size, learning rate, number of EPOCH, etc.

## Dataset

- `X.txt`: Contains the text of scientific abstracts.
- `YL1.txt`: Contains the integer labels (0-6) for each abstract.

### Categories

- 0: Computer Science
- 1: Electrical Engineering
- 2: Psychology
- 3: Mechanical Engineering
- 4: Civil Engineering
- 5: Medical Science
- 6: Biochemistry

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (optional but recommended for faster training)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/AnhBui1108/BERT-Fine-Tunning-For-Document-Classification.git
   cd bert-fine-tuning
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Open the Jupyter notebook and run all cells:

```
jupyter notebook BERT-fine-tuning.ipynb
```

## Key Learnings

- **Fine-tuning vs. Pre-training**: Adapted pre-trained models to downstream tasks using limited task-specific data, reducing training time and compute compared with training from scratch..
- **Tokenization**: Handled subword tokenization and special tokens (`[CLS]`, `[SEP]`).
- **Optimization**: Explored how hyperparameters such as batch size and learning rate affect training stability and convergence.

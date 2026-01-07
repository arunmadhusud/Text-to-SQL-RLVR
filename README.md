# SQL RLVR

Fine-tuning a language model for Text-to-SQL using a two-stage approach: Supervised Fine-Tuning (SFT) followed by Reinforcement Learning from Verifiable Rewards (RLVR).

## Overview

**Task**: Convert natural language questions to SQL queries.

**Approach**:
1. **Stage 1 - SFT**: Fine-tune a base model using LoRA on question-SQL pairs from the Spider dataset.
2. **Stage 2 - RLVR**: Further train the LoRA adapter using GRPO (Group Relative Policy Optimization) with execution-based rewards. The reward function executes predicted SQL against the database and compares results with the gold query.

**Model**: [Qwen3-4B-Instruct](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507)

**Dataset**: [Spider](https://huggingface.co/datasets/xlangai/spider) - A large-scale complex text-to-SQL benchmark with 7,000 training examples across multiple databases.

## Training

### Reward Function

The RLVR stage uses an execution-based reward that:
- Executes predicted SQL on the actual database
- Compares results with gold query execution
- Assigns rewards based on result matching:
  - `+5.0` for exact match
  - `-1.0` for invalid SQL
  - `-2.0` for empty response
  - Partial credit based on F1 score for partial matches

## Installation

```bash
pip install unsloth vllm python-dotenv
```

## Setup

1. Copy `.env.example` to `.env` and add your WandB API key:
```bash
cp .env.example .env
```

2. Download Spider database files:
```bash
gdown 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J
unzip spider_data.zip -d spider_data
```

3. Update paths in `config.py` to match your setup.

## Usage

### 1. SFT Training
```bash
python train_sft.py
```

### 2. GRPO Training
```bash
python train_grpo.py
```

### 3. Evaluation
```bash
python evaluate.py
```

This generates `pred.txt` and `gold.txt` files for evaluation.

## Project Structure

```
├── .env.example      # Template for API keys
├── config.py         # Paths and hyperparameters
├── utils.py          # Model loading, data prep, rewards
├── train_sft.py      # Supervised fine-tuning
├── train_grpo.py     # GRPO/RLVR training
└── evaluate.py       # Batch inference on validation set
```

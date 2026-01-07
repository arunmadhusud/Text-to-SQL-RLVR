import os

# Paths
SPIDER_PATH = "/home/ubuntu/sqlsftrlvr/spider_data/spider_data"
SPIDER_DB_PATH = f"{SPIDER_PATH}/database"
SFT_OUTPUT_DIR = "./checkpoints/sft"
GRPO_OUTPUT_DIR = "./checkpoints/grpo"

# Model
MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32

# SFT Training
SFT_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "num_train_epochs": 1,
    "learning_rate": 2e-4,
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.001,
    "lr_scheduler_type": "linear",
    "seed": 3407,
}

# GRPO Training
GRPO_CONFIG = {
    "learning_rate": 5e-6,
    "weight_decay": 0.001,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "linear",
    "optim": "adamw_8bit",
    "logging_steps": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "num_generations": 4,
    "max_prompt_length": 1074,
    "max_completion_length": 512,
    "max_steps": 100,
}

# Wandb
WANDB_PROJECT = "sql_rlvr"
WANDB_ENTITY = "arunmadhusudh-northeastern-university"

# System prompt
SYSTEM_PROMPT = "Convert natural language to SQL. Output only valid SQL."
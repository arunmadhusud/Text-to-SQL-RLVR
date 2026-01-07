import os
import sqlite3
import numpy as np
import wandb
from collections import Counter
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_data_formats
from config import (
    MODEL_NAME, MAX_SEQ_LENGTH, LORA_RANK, 
    WANDB_PROJECT, WANDB_ENTITY, SPIDER_DB_PATH, SYSTEM_PROMPT
)

# ============== Model Loading ==============

def load_model_for_training(fast_inference=False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=fast_inference,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.9,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    return model, tokenizer


def load_model_with_adapter(adapter_path, fast_inference=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=fast_inference,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.9,
    )
    model.load_adapter(adapter_path)
    return model, tokenizer


def init_wandb(run_name, stage, extra_config=None):
    config = {
        "model": MODEL_NAME,
        "stage": stage,
        "dataset": "spider",
    }
    if extra_config:
        config.update(extra_config)
    
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=config,
    )


# ============== Data ==============

def get_schema_from_db(db_id):
    db_path = os.path.join(SPIDER_DB_PATH, db_id, f"{db_id}.sqlite")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT sql FROM sqlite_master 
        WHERE type='table' AND sql IS NOT NULL
        ORDER BY name
    """)
    tables = cursor.fetchall()
    conn.close()
    schema = "\n\n".join(t[0] for t in tables if t[0])
    return schema


def load_spider_dataset():
    dataset = load_dataset("xlangai/spider")
    train_dataset = standardize_data_formats(dataset["train"])
    valid_dataset = standardize_data_formats(dataset["validation"])
    return train_dataset, valid_dataset


def convert_to_conversations(examples):
    conversations_list = []
    for question, query, db_id in zip(examples["question"], examples["query"], examples["db_id"]):
        schema = get_schema_from_db(db_id)
        user_content = f"""Database: {db_id}

Schema:
{schema}

Question: {question}"""
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": query}
        ]
        conversations_list.append(conversation)
    return {"conversations": conversations_list}


def prepare_sft_dataset(tokenizer):
    train_dataset, valid_dataset = load_spider_dataset()
    
    train_dataset = train_dataset.map(
        convert_to_conversations,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    valid_dataset = valid_dataset.map(
        convert_to_conversations,
        batched=True,
        remove_columns=valid_dataset.column_names
    )
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, 
                tokenize=False, 
                add_generation_prompt=False
            ) 
            for convo in convos
        ]
        return {"text": texts}
    
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    valid_dataset = valid_dataset.map(formatting_prompts_func, batched=True)
    
    return train_dataset, valid_dataset


def prepare_grpo_dataset(example):
    schema = get_schema_from_db(example["db_id"])
    user_content = f"""Database: {example["db_id"]}

Schema:
{schema}

Question: {example["question"]}"""
    
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "answer": example["query"],
        "db_id": example["db_id"],
    }


def prepare_grpo_datasets(tokenizer):
    train_dataset, valid_dataset = load_spider_dataset()
    
    train_dataset = train_dataset.map(prepare_grpo_dataset)
    valid_dataset = valid_dataset.map(prepare_grpo_dataset)
    
    tokenized = train_dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True,
    )
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    
    lengths = tokenized["L"]
    print(f"Min: {min(lengths)}, Mean: {np.mean(lengths):.0f}, Max: {max(lengths)}")
    print(f"90th percentile: {np.percentile(lengths, 90):.0f}")
    
    max_length = int(np.percentile(lengths, 90))
    print(f"Filtering to max length: {max_length}")
    print(f"Examples before: {len(train_dataset)}")
    
    train_dataset = train_dataset.select(np.where(np.array(tokenized["L"]) <= max_length)[0])
    print(f"Examples after: {len(train_dataset)}")
    
    tokenized_val = valid_dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True,
    )
    tokenized_val = tokenized_val.map(lambda x: {"L": len(x["tokens"])})
    valid_dataset = valid_dataset.select(np.where(np.array(tokenized_val["L"]) <= max_length)[0])
    print(f"Validation examples after filter: {len(valid_dataset)}")
    
    return train_dataset, valid_dataset


# ============== Rewards ==============

PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5


def execute_sql(sql, db_id):
    db_path = os.path.join(SPIDER_DB_PATH, db_id, f"{db_id}.sqlite")
    try:
        conn = sqlite3.connect(db_path, timeout=1.0)
        cursor = conn.cursor()
        cursor.execute("PRAGMA query_only = TRUE")
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return None, str(e)


def normalize_results(rows):
    return Counter([tuple(r) for r in rows])


def sql_execution_reward(prompts, completions, answer, db_id, **kwargs):
    global PRINTED_TIMES, PRINT_EVERY_STEPS
    
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    scores = []

    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print("*" * 20)
        print(f"Question: {question}")
        print(f"DB: {db_id[0]}")
        print(f"Gold SQL: {answer[0]}")
        print(f"Predicted: {responses[0]}")
    PRINTED_TIMES += 1

    for pred_sql, gold_sql, db in zip(responses, answer, db_id):
        if not pred_sql:
            scores.append(-2.0)
            continue

        pred_sql_clean = pred_sql.strip()
        pred_results, pred_err = execute_sql(pred_sql_clean, db)
        gold_results, gold_err = execute_sql(gold_sql, db)

        if pred_err or pred_results is None:
            scores.append(-1.0)
            continue

        if gold_err or gold_results is None:
            scores.append(0.0)
            continue

        pred_norm = normalize_results(pred_results)
        gold_norm = normalize_results(gold_results)

        if pred_norm == gold_norm:
            scores.append(5.0)
            continue

        if len(gold_norm) == 0:
            reward = 0.0 if len(pred_norm) == 0 else -0.5
            scores.append(reward)
            continue

        tp = sum((pred_norm & gold_norm).values())
        fp = sum((pred_norm - gold_norm).values())
        fn = sum((gold_norm - pred_norm).values())

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        reward = -0.5 + 2.0 * f1
        scores.append(reward)

    return scores
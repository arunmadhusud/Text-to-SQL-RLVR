import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from dotenv import load_dotenv
load_dotenv()

import wandb
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer

from config import GRPO_CONFIG, GRPO_OUTPUT_DIR, SFT_OUTPUT_DIR
from utils import prepare_grpo_datasets, sql_execution_reward, load_model_with_adapter, init_wandb


def main():
    # Load model with SFT adapter
    sft_checkpoint = f"{SFT_OUTPUT_DIR}/checkpoint-875"
    model, tokenizer = load_model_with_adapter(sft_checkpoint, fast_inference=True)
    
    train_dataset, valid_dataset = prepare_grpo_datasets(tokenizer)
    
    init_wandb("grpo-spider-qwen3-4b", "GRPO", GRPO_CONFIG)
    
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    
    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=GRPO_CONFIG["learning_rate"],
        weight_decay=GRPO_CONFIG["weight_decay"],
        warmup_ratio=GRPO_CONFIG["warmup_ratio"],
        lr_scheduler_type=GRPO_CONFIG["lr_scheduler_type"],
        optim=GRPO_CONFIG["optim"],
        logging_steps=GRPO_CONFIG["logging_steps"],
        per_device_train_batch_size=GRPO_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=GRPO_CONFIG["gradient_accumulation_steps"],
        num_generations=GRPO_CONFIG["num_generations"],
        max_prompt_length=GRPO_CONFIG["max_prompt_length"],
        max_completion_length=GRPO_CONFIG["max_completion_length"],
        max_steps=GRPO_CONFIG["max_steps"],
        output_dir=GRPO_OUTPUT_DIR,
        report_to="wandb",
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[sql_execution_reward],
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    
    wandb.finish()


if __name__ == "__main__":
    main()
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from dotenv import load_dotenv
load_dotenv()

import torch
import wandb
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

from config import SFT_CONFIG, SFT_OUTPUT_DIR
from utils import prepare_sft_dataset, load_model_for_training, init_wandb


def main():
    model, tokenizer = load_model_for_training(fast_inference=True)
    train_dataset, valid_dataset = prepare_sft_dataset(tokenizer)
    
    print(train_dataset[0]["text"])
    
    init_wandb("sft-spider-qwen3-4b", "SFT", SFT_CONFIG)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=SFT_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=SFT_CONFIG["gradient_accumulation_steps"],
            warmup_steps=SFT_CONFIG["warmup_steps"],
            num_train_epochs=SFT_CONFIG["num_train_epochs"],
            learning_rate=SFT_CONFIG["learning_rate"],
            logging_steps=SFT_CONFIG["logging_steps"],
            optim=SFT_CONFIG["optim"],
            weight_decay=SFT_CONFIG["weight_decay"],
            lr_scheduler_type=SFT_CONFIG["lr_scheduler_type"],
            seed=SFT_CONFIG["seed"],
            report_to="wandb",
            output_dir=SFT_OUTPUT_DIR,
        ),
    )
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    
    # Verify masking
    print(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
    print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " "))
    
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    trainer_stats = trainer.train()
    
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    
    wandb.finish()


if __name__ == "__main__":
    main()
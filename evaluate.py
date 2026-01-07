import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm
from datasets import load_dataset
from vllm import SamplingParams

from config import SFT_OUTPUT_DIR, SYSTEM_PROMPT
from utils import get_schema_from_db, load_model_for_training


def main():
    model, tokenizer = load_model_for_training(fast_inference=True)
    
    spider_dev = load_dataset("xlangai/spider", split="validation")
    
    print("Preparing prompts...")
    all_texts = []
    golds = []
    
    for example in tqdm(spider_dev):
        schema = get_schema_from_db(example["db_id"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Database: {example['db_id']}\n\nSchema:\n{schema}\n\nQuestion: {example['question']}"}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        all_texts.append(text)
        golds.append(f"{example['query']}\t{example['db_id']}")
    
    print("Running vLLM batch inference...")
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=256,
    )
    
    sft_checkpoint = f"{SFT_OUTPUT_DIR}/checkpoint-875"
    outputs = model.fast_generate(
        all_texts,
        sampling_params=sampling_params,
        lora_request=model.load_lora(sft_checkpoint)
    )
    
    predictions = []
    for output in outputs:
        pred_sql = output.outputs[0].text.strip().split('\n')[0]
        predictions.append(pred_sql)
    
    with open("pred.txt", "w") as f:
        f.write("\n".join(predictions))
    
    with open("gold.txt", "w") as f:
        f.write("\n".join(golds))
    
    print(f"Files saved! Generated {len(predictions)} predictions.")


if __name__ == "__main__":
    main()
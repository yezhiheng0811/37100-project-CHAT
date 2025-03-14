#%%
import torch
import evaluate
from typing import List, Dict
from datasets import load_dataset
import numpy as np

# Transformers / PEFT / BitsAndBytes
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback,
    logging,
)

import copy

logging.set_verbosity_error()

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# 1. Callback to empty cache after each training step
class EmptyCacheCallback(TrainerCallback):
    def on_train_batch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        return control

# 2. BitsAndBytesConfig for 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# 3. Load base model in 4-bit and prepare for LoRA
model_name = "meta-llama/Llama-3.1-8B"
print(f"Loading {model_name} with 4-bit quantization...")

model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
)

baseline_model = copy.deepcopy(model_4bit)

model_4bit = prepare_model_for_kbit_training(model_4bit)

# 4. LoRA Configuration
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=4,                
    lora_alpha=16,      
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model_4bit, lora_config)
model.config.use_cache = False  

model.print_trainable_parameters()

# 5. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 6. Data Preprocessing
max_length = 2048  

def collate_fn(examples):
    ref_max_len = 300
    q_max_len   = 50
    a_max_len   = 200

    all_input_ids = []

    refs_list = examples["references"]
    ques_list = examples["question"]
    ans_list  = examples["answer"]

    for i in range(len(refs_list)):
        references = refs_list[i]
        question   = ques_list[i]
        answer     = ans_list[i]

        if isinstance(references, list):
            references = " ".join(str(r) for r in references)
        if isinstance(question, list):
            question = " ".join(str(q) for q in question)
        if isinstance(answer, list):
            answer = " ".join(str(a) for a in answer)

        ref_enc = tokenizer(
            references,
            truncation=True, 
            max_length=ref_max_len, 
            add_special_tokens=False
        )
        q_enc = tokenizer(
            question, 
            truncation=True, 
            max_length=q_max_len, 
            add_special_tokens=False
        )
        a_enc = tokenizer(
            answer, 
            truncation=True, 
            max_length=a_max_len, 
            add_special_tokens=False
        )

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

        final_ids = []
        if bos_id is not None:
            final_ids.append(bos_id)

        final_ids.extend(ref_enc["input_ids"])
        if eos_id is not None:
            final_ids.append(eos_id)

        final_ids.extend(q_enc["input_ids"])
        if eos_id is not None:
            final_ids.append(eos_id)

        final_ids.extend(a_enc["input_ids"])
        if eos_id is not None:
            final_ids.append(eos_id)

        all_input_ids.append(final_ids)

    padded = tokenizer.pad(
        {"input_ids": all_input_ids},
        padding=True,
        return_tensors="pt"
    )

    padded["labels"] = padded["input_ids"].clone()
    return padded

# 7. Load and Tokenize Dataset 
dataset = load_dataset("THUDM/webglm-qa")
print("Subsetting dataset to reduce memory usage...")
dataset["train"] = dataset["train"].select(range(1000))   
dataset["validation"] = dataset["validation"].select(range(100))

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    collate_fn,
    batched=True,     
    batch_size=8,
    desc="Tokenizing dataset"
)

keep_cols = ["input_ids", "attention_mask", "labels"]
def filter_cols(ds):
    drop_cols = set(ds.column_names) - set(keep_cols)
    return ds.remove_columns(list(drop_cols))

for split in tokenized_dataset.keys():
    tokenized_dataset[split] = filter_cols(tokenized_dataset[split])

# 8. Minimal Data Collator
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        padded_features = []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            new_f = {
                "input_ids": f["input_ids"] + [self.tokenizer.pad_token_id]*pad_len,
                "attention_mask": f["attention_mask"] + [0]*pad_len,
                "labels": f["labels"] + [-100]*pad_len
            }
            padded_features.append(new_f)

        batch = {}
        for k in padded_features[0].keys():
            arr = [pf[k] for pf in padded_features]
            batch[k] = torch.tensor(arr, dtype=torch.long)
        return batch

# 9. Define a Metric and Compute Fn
bleu = evaluate.load("bleu")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    decoded_labels = []
    for label_seq in labels:
        label_seq = [l for l in label_seq if l != -100]
        decoded_labels.append(tokenizer.decode(label_seq, skip_special_tokens=True))

    references = [[ref] for ref in decoded_labels]
    results = bleu.compute(predictions=decoded_preds, references=references)
    return {"bleu": results["bleu"]}

def evaluate_baseline_model(model, dataset, tokenizer, data_collator, batch_size=4):
    model.eval()

    all_decoded_preds = []
    all_decoded_labels = []

    val_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    for batch in val_dataloader:
        for k in batch:
            batch[k] = batch[k].to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits  

        preds = logits.argmax(dim=-1) 
        preds_cpu = preds.detach().cpu().numpy()
        labels_cpu = batch["labels"].detach().cpu().numpy()

        for i in range(preds_cpu.shape[0]):
            pred_ids = preds_cpu[i]  
            decoded_pred = tokenizer.decode(pred_ids, skip_special_tokens=True)
            all_decoded_preds.append(decoded_pred)

            label_ids = labels_cpu[i]
            label_ids = [l for l in label_ids if l != -100]
            decoded_label = tokenizer.decode(label_ids, skip_special_tokens=True)
            all_decoded_labels.append(decoded_label)

    references = [[ref] for ref in all_decoded_labels]
    results = bleu.compute(predictions=all_decoded_preds, references=references)
    return results["bleu"]

# 10. Training Arguments with Evaluation
training_args = TrainingArguments(
    output_dir="./lora_finetuned_model",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=1e-4,            
    logging_steps=50,
    eval_accumulation_steps=4,
    eval_strategy="epoch",    
    save_steps=200,
    save_total_limit=1,
    fp16=True,                      
    remove_unused_columns=False,
    gradient_checkpointing=True,    
    report_to="none",
)

# Evaluate baseline model 
print("Evaluating baseline 4-bit model (no LoRA) on validation set...")
baseline_bleu = evaluate_baseline_model(
    model=baseline_model,
    dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=CustomDataCollator(tokenizer),
    batch_size=4 
)
print(f"Baseline BLEU: {baseline_bleu}")

# 11. Trainer with compute_metrics and callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=CustomDataCollator(tokenizer),
    callbacks=[EmptyCacheCallback()],
    compute_metrics=compute_metrics
)

# 12. Train
trainer.train()

# Evaluate fine-tuned model and compare BLEU scores
finetuned_metrics = trainer.evaluate()
finetuned_bleu = finetuned_metrics['eval_bleu']
print(f"Fine-tuned BLEU: {finetuned_bleu}")
improvement = finetuned_bleu - baseline_bleu
print(f"BLEU improvement: {improvement}")

# 13. Save Model
trainer.save_model("./lora_finetuned_model")
tokenizer.save_pretrained("./lora_finetuned_model")
print("LoRA fine-tuning complete. Model saved to ./lora_finetuned_model.")

# 14. Simple Inference
model.eval()

def generate_answer(question):
    prompt = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=1,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Pick the first 2 questions from the validation set
val_data = dataset["validation"]
for i in range(2):
    question = val_data[i]["question"]
    print("\nDataset Question:", question)
    generated = generate_answer(question)
    print("Model's Generated Answer:", generated)

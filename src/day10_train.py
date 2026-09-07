from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

model_name = "Qwen/Qwen3-0.6B"

print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files="data/daa_train.jsonl",
    split="train"
)

print(dataset)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def format_example(example):
    return (
        "Question: "
        + example["question"]
        + "\nAnswer: "
        + example["answer"]
    )

def tokenize_example(example):
    text = format_example(example)

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=512
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

print("\nTokenizing dataset...")

tokenized_dataset = dataset.map(
    tokenize_example
)

model = AutoModelForCausalLM.from_pretrained(
    model_name
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(
    model,
    lora_config
)

print("\nLoRA parameters:")
model.print_trainable_parameters()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="outputs/day10_test",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("\nStarting training...")

trainer.train()

print("\nSaving LoRA adapter...")

model.save_pretrained(
    "outputs/day10_test"
)

tokenizer.save_pretrained(
    "outputs/day10_test"
)

print("\nTraining complete.")
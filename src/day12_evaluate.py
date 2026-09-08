import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


MODEL_NAME = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "outputs/day10_test"


# --------------------------------------------------
# Device
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# Load original/base model
# --------------------------------------------------

print("Loading base Qwen model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

base_model = base_model.to(device)
base_model.eval()


# --------------------------------------------------
# Load fine-tuned model
# --------------------------------------------------

print("Loading LoRA model...")

finetuned_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

finetuned_model = PeftModel.from_pretrained(
    finetuned_model,
    ADAPTER_PATH
)

finetuned_model = finetuned_model.to(device)
finetuned_model.eval()


# --------------------------------------------------
# Load evaluation dataset
# --------------------------------------------------

print("Loading evaluation dataset...")

dataset = load_dataset(
    "json",
    data_files="data/daa_eval.jsonl",
    split="train"
)


# --------------------------------------------------
# Generation function
# --------------------------------------------------

def generate_answer(model, question):

    prompt = (
        "Question: "
        + question
        + "\nAnswer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            repetition_penalty=1.1
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

for example in dataset:

    question = example["question"]

    print("\n" + "=" * 70)

    print("QUESTION:")
    print(question)

    print("\nBASE QWEN:")
    base_answer = generate_answer(
        base_model,
        question
    )
    print(base_answer)

    print("\nFINE-TUNED QWEN:")
    finetuned_answer = generate_answer(
        finetuned_model,
        question
    )
    print(finetuned_answer)
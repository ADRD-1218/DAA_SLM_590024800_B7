import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "Qwen/Qwen3-0.6B"
adapter_path = "outputs/day10_test"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    adapter_path
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)

print("Using device:", device)

questions = [
    "What is the time complexity of binary search?",
    "What is linear search?",
    "What is a minimum spanning tree?",
    "Compare BFS and DFS."
]

for question in questions:

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

    print("\n" + "=" * 60)
    print("Question:", question)
    print("Generating answer...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            repetition_penalty=1.1
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    print("Answer:", answer)
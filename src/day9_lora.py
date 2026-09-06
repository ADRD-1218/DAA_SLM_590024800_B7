from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

model_name = "Qwen/Qwen3-0.6B"

model = AutoModelForCausalLM.from_pretrained(model_name)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Before LoRA")
print("Total parameters:", total_parameters)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

print("\nAfter LoRA")
model.print_trainable_parameters()

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nTrainable parameter count:")
print(trainable_parameters)

print("\nTrainable percentage:")
print(
    f"{trainable_parameters / total_parameters * 100:.4f}%"
)
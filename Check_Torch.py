import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch CUDA version:", torch.version.cuda)

    gpu_properties = torch.cuda.get_device_properties(0)

    print(
        "VRAM:",
        round(gpu_properties.total_memory / (1024 ** 3), 2),
        "GB"
    )
else:
    print("CUDA is NOT available.")
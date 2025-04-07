import torch

print(torch.cuda.is_available())  # Prints True if CUDA is available, otherwise False
import torch

if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")

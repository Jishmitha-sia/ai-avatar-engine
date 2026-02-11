import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"Is CUDA (GPU) available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not found. The system will use your CPU instead.")
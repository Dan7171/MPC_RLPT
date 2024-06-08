import torch

# Check if CUDA is available
while True:
    if torch.cuda.is_available():
        # Create tensors on GPU
        device = torch.device("cuda")
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        # Perform matrix multiplication on GPU
        c = torch.matmul(a, b)
        
        # Move result back to CPU for printing
        c_cpu = c.cpu()
        
        print("Matrix multiplication on GPU successful!")
    else:
        print("CUDA is not available. Make sure you have CUDA enabled GPU and PyTorch with CUDA support installed.")
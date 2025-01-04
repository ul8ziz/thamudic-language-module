import torch
import sys

def check_gpu():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Test CUDA device
        x = torch.rand(3, 3).cuda()
        print("\nTesting CUDA device with tensor operations:")
        print(x)
        print(f"Tensor device: {x.device}")
    else:
        print("\nNo CUDA device available. Please check your PyTorch installation.")
        print("You may need to reinstall PyTorch with CUDA support:")
        print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    check_gpu()

import torch
def cudaCheck():
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} name:", torch.cuda.get_device_name(i))
            print(f"GPU {i} capability:", torch.cuda.get_device_capability(i))
        
        # Force current device
        torch.cuda.set_device(0)
        print("Current device:", torch.cuda.current_device())
        print("Device being used:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("WARNING: CUDA is not available. Training will be on CPU only.")
        print("If you have a GPU, please check your PyTorch installation.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device
# verify_gpu.py

import torch

print("--- GPU Verification Script ---")
if torch.cuda.is_available():
    print("SUCCESS: PyTorch can see your GPU.")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version (PyTorch built with): {torch.version.cuda}")
    print("\nRunning a quick test...")
    try:
        a = torch.tensor([1.0, 2.0]).to("cuda")
        b = a * 2
        print(f"Test calculation (should be [2., 4.]): {b}")
        print("\nYour setup is correct and ready for training!")
    except Exception as e:
        print(f"\nERROR: GPU test failed. {e}")
else:
    print("FAILURE: PyTorch cannot see your GPU.")
    print("Please ensure your NVIDIA drivers are up to date and you installed the correct PyTorch CUDA build.")

print("--- End of Script ---")
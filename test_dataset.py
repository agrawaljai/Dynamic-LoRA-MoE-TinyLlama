# test_dataset.py
from datasets import load_dataset

print("--- Attempting to download the dataset... ---")
try:
    ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split='train[:1%]')
    print("\n\nSUCCESS: Dataset 'HuggingFaceH4/CodeAlpaca_20K' was found and downloaded.")
    print("\nHere is a sample row:")
    print(ds[0])
except Exception as e:
    print("\n\nFAILURE: The dataset could not be loaded.")
    print(f"Error: {e}")
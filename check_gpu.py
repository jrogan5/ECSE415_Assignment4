#!/usr/bin/env python3
"""
Quick script to check GPU availability for PyTorch
"""

import torch
import sys

print("=" * 70)
print("GPU AVAILABILITY CHECK")
print("=" * 70)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\n✓ CUDA Available: {cuda_available}")

if cuda_available:
    # Get GPU details
    device_count = torch.cuda.device_count()
    print(f"✓ Number of GPUs: {device_count}")

    for i in range(device_count):
        print(f"\n  GPU {i}:")
        print(f"    Name: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")

    print(f"\n✓ Recommended device setting: device=0")
else:
    print(f"✗ Number of GPUs: 0")
    print(f"\n⚠️  No CUDA GPU detected!")
    print(f"✓ Recommended device setting: device='cpu'")
    print(f"\nNote: Training will be slower on CPU but will still work.")

# Check PyTorch version
print(f"\n{'=' * 70}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version (if available): {torch.version.cuda if cuda_available else 'N/A'}")
print("=" * 70)

# Provide recommendation
print("\n📋 RECOMMENDATION:")
if cuda_available:
    print("  Your system has GPU(s) available. Use device=0 in your code.")
else:
    print("  Your system does NOT have CUDA GPU. Change device=0 to device='cpu'")
    print("  Or use device='cuda' if torch.cuda.is_available() else 'cpu' for auto-detection")

sys.exit(0 if cuda_available else 1)

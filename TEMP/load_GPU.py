import torch
import time
import os

assert torch.cuda.is_available(), "CUDA not available"

device = torch.device("cuda:0")  # 映射后的 cuda:0 实际是物理 GPU:4
torch.cuda.set_device(device)

# -----------------------------
# A100 80G 显存估算
# FP16: 2 bytes
# 120000 x 120000 x 2 bytes ≈ 26.8 GB
# 3 个常驻张量 ≈ 80GB
# -----------------------------

N = 40_000
dtype = torch.float16

print("Allocating large tensors...")

A = torch.randn((N, N), device=device, dtype=dtype)
B = torch.randn((N, N), device=device, dtype=dtype)
C = torch.empty((N, N), device=device, dtype=dtype)

torch.cuda.synchronize()
print("Allocation done.")
print(f"Allocated ~{3 * N * N * 2 / 1024**3:.1f} GB")

# Warm-up
for _ in range(3):
    C = torch.matmul(A, B)
torch.cuda.synchronize()

print("Entering main loop (Ctrl+C to stop)...")

# -----------------------------
# 主循环：持续 GEMM
# 保证 GPU 利用率
# -----------------------------
while True:
    start = time.time()

    # 多次 matmul 提高算力占用
    for _ in range(5):
        C = torch.matmul(A, B)

    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Iteration time: {elapsed:.2f}s")

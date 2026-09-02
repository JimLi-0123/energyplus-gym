import torch

# 檢查是否有可用的 GPU
print(torch.cuda.is_available())  # 如果顯示 True，表示可以使用 GPU，否則是 CPU
import torch
print(torch.cuda.is_available())
# gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
# gpu_memory_gb = round(gpu_memory_bytes / 2**30)
# print(f"Available memory in GB: {gpu_memory_gb} GB")

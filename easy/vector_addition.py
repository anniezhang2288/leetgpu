import torch

# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    return C.copy_(A+B)



# # A, B, C are tensors on the GPU
# def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
#     torch.add(A,B,alpha=1,out=C)
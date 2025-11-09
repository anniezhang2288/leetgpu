import torch


# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    
    output_len = input_size - kernel_size + 1
    unfolded_input = input.unfold(dimension=0, size=kernel_size, step=1)
    output.copy_(unfolded_input @ kernel)
    
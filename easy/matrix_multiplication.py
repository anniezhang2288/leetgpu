import torch

# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):

    torch.matmul(A, B, out = C)

    # row * column
    # check if it is valid
    # m = 1 n = 3 k = 1
    # for row in m
    # get the row 
    # block_size = 64

    # for row_idx in range(0, M, block_size): # grab each row of A
    #     row_end  = min(row_idx + block_size, M)
        
    #     for col_idx in range(0, K, block_size): # grab each column of B
    #         col_end  = min(col_idx + block_size, K)

    #         sum = torch.zeros(row_end - row_idx, col_end - col_idx, device = 'cuda')
    #         for k in range(0, N, block_size):
    #             k_end =  min(k + block_size, N)
    #             # print("k and k_end:", k, k_end)
    #             # print("a:", A[row_idx: row_end,k:k_end])
    #             # print("b",  B[k:k_end, col_idx:col_end])
    #             sum += A[row_idx: row_end,k:k_end] @ B[k:k_end, col_idx:col_end]
            
    #         C[row_idx:row_end,col_idx: col_end] = sum
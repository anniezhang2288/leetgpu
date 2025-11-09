import torch

# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    ## First solution, times out
    # subtracting = torch.tensor([255]*width*height*4).to(device='cuda')
    # for idx in range(3, len(image), 4):
    #     subtracting[idx] = 2*image[idx]


    # image.copy_(torch.sub(subtracting, image))


    # [255]*width*height*4 creates a giant list on the CPU,
    # python must individual box those intergers and then torch.tensor()
    # converts this list to a CPU tensor which is then transfered to the GPU
    # due to .to('cuda')


    ## Learnings: use torch.full() or torch.full_like
    # directly on the GPU

    # Solution 2: 
    # makes a torch tensor of size (width*height*4,) filled with 255
    subtracting = torch.full((width*height*4,), 255, device = 'cuda')
    subtracting[3::4] = 2 * image[3::4]
    image.sub_(subtracting).neg_()
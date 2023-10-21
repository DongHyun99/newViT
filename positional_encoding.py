import torch

def sinusoidal_PE_2d(num_patches_2d, dim):
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(dim))
        
    pe = torch.zeros(num_patches_2d.size()[0], dim, device=0)
    denominators = torch.pow(10000.0, 2*torch.arange(0, dim, 2)/dim)
    pe[:, 0::2] = torch.sin(num_patches_2d[:,0].unsqueeze(1).float() / denominators)
    pe[:, 1::2] = torch.cos(num_patches_2d[:,1].unsqueeze(1).float() / denominators)
    pe = pe.unsqueeze(0)

    return pe
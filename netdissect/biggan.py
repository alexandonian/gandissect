import torch

from pretorched.gans import BigGAN


def fix_class(G, y):
    f = G.forward

    def forward(self, z):
        bs = z.size(0)
        c = y * torch.ones(bs, device=z.device).long()
        return f(z, c, embed=True)

    setattr(G.__class__, 'forward', forward)
    return G


def pretrained(res=128, pretrained='places365', y=215):
    G = BigGAN(resolution=res, pretrained=pretrained)
    G = fix_class(G, y)
    return G


def print_network(net, verbose=False):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        print(net)
    print(f'Total number of parameters: {(num_params / 1e6):3.3f} M')


def from_pth_file(filename):
    """Instantiate from a pth file."""

    state_dict = torch.load(filename)
    return state_dict


def from_weight_dir(weight_dir):
    pass

import torch
import numpy as np
from scipy.linalg import sqrtm


def gradient_penalty(D, x_real, x_fake, derivative, device):
    '''gradient penalty that is used in https://arxiv.org/abs/1704.00028'''
    if len(x_real.shape) == 2:
        alpha = torch.rand(x_real.size()[0], 1, device=device)
    else:
        alpha = torch.rand(x_real.size()[0], 1, 1, 1, device=device)

    alpha = alpha.expand(x_real.size())
    interpolates = alpha * x_real + (1-alpha) * x_fake
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = D(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1)-derivative) ** 2).mean() * 10
    return gradient_penalty


def nn_accuracy(p_real, p_fake, device="cpu", k=5):
    size = p_fake.shape[0]
    p_real = p_real.reshape(size, -1).to(device)
    p_fake = p_fake.reshape(size, -1).to(device)
    p_all = torch.cat([p_fake, p_real], dim=0)
    dists = torch.cdist(p_all, p_all) + torch.eye(2*size, device=device) * 1e12
    values, indexes = torch.topk(dists, k=k, largest=False)

    decisions = (indexes > size-1).sum(dim=1).float() / k
    real_acc = (decisions[size:].sum()).float() / size
    fake_acc = (size - decisions[:size].sum()).float() / size

    return real_acc.item(), fake_acc.item()


def FID_score(x_real, x_fake):
    mu_real = x_real.mean(dim=0)
    mu_fake = x_fake.mean(dim=0)
    cov_real = np.cov(x_real, rowvar=False)
    cov_fake = np.cov(x_fake, rowvar=False)
    mu_diff = np.linalg.norm(mu_real-mu_fake, 2) ** 2
    covmean = 2 * sqrtm(np.matmul(cov_real, cov_fake.T))
    cov_diff = np.trace(cov_real + cov_fake - covmean)
    return mu_diff + cov_diff.real


def get_parameter_count(model):
    total_num = 0
    for p in model.parameters():
        total_num += p.shape.numel()

    return total_num

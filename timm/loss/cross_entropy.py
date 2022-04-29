""" Cross Entropy w/ smoothing or soft targets

Hacked together by / Copyright 2021 Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

<<<<<<< HEAD

# @torch.jit.script
def logsoftmultimaxselect(x, groups, groups_all):
    assert len(x.shape) <= 2
    x = x - x.max(1, keepdim=True)[0]
    # nums = torch.zeros(x.shape[0], dtype=x.dtype)
    nums = []
    # for k, group in enumerate(self.targets_hot_loc[1:]):
    for k, group in enumerate(groups):
        nums.append(torch.exp(x[k, group]).sum())
        # nums[k] = torch.exp(x[k, group]).sum()
    denoms = torch.exp(x[:, groups_all[0]]).sum(dim=1)
    for group in groups_all[1:]:
        denoms += torch.exp(x[:, group]).sum(dim=1)
    out = torch.log(torch.tensor(nums, device=x.device)) - torch.log(denoms)
    # breakpoint()
    # nums = torch.tensor([torch.exp(x[k, group]).sum() # Causes error with torch.jit
                        # for k, group in enumerate(groups)], device=x.device)
    # nums = torch.empty(len(groups), device=x.device)
    # temp = torch.exp(x[0, groups[0]])
    # print(temp)
    # temp2 = torch.sum(temp)
    # print(temp2)
    # nums[0] = temp2
    # for k, group in groups:
        # temp = torch.exp(x[k, group])
        # # print(temp)
        # temp2 = torch.sum(temp)
        # # print(temp2)
        # nums[k] = temp2
    # breakpoint()
    # denoms = 0
    # for group in groups_all:
        # denoms += torch.exp(x[:, group]).sum(dim=1)
    # denoms = torch.stack([torch.exp(x[:, group]).sum(dim=1)
                           # for group in groups_all]).sum(dim=1)
    # out = torch.log(nums) - torch.log(denoms)
    return out

def logsoftmultimax(x, groups_all):
    assert len(x.shape) <= 2
    x = x - x.max(1, keepdim=True)[0]
    # breakpoint()
    nums = torch.tensor([torch.exp(x[k, group]).sum() # Causes error with torch.jit
                        for k, group in enumerate(groups)], device=x.device)
    # nums = torch.empty(len(groups), device=x.device)
    # temp = torch.exp(x[0, groups[0]])
    # print(temp)
    # temp2 = torch.sum(temp)
    # print(temp2)
    # nums[0] = temp2
    # for k, group in groups:
        # temp = torch.exp(x[k, group])
        # # print(temp)
        # temp2 = torch.sum(temp)
        # # print(temp2)
        # nums[k] = temp2
    # breakpoint()
    denoms = 0
    for group in groups_all:
        denoms += torch.exp(x[:, group]).sum(dim=1)
    # denoms = torch.stack([torch.exp(x[:, group]).sum(dim=1)
                           # for group in groups_all]).sum(dim=1)
    out = torch.log(nums) - torch.log(denoms)
    return out


class MultihotCrossEntropy(nn.Module):

    def __init__(self, n_classes, target_dim, n_hot, agg_fun=None, rng=None):
        """
        
        Parameters
        ----------
        rng : Optional[torch.Generator]
        """
        super().__init__()
        self.targets_hot_loc = nn.Parameter(
            torch.empty(n_classes, n_hot, dtype=torch.long),
            requires_grad=False)
        self.targets = nn.Parameter(
            torch.zeros(n_classes, target_dim, dtype=torch.long),
            requires_grad=False)
        for c in range(n_classes):
            loc = torch.randperm(target_dim, generator=rng)[:n_hot]
            self.targets_hot_loc[c] = loc
            self.targets[c, loc] = 1
        # self.agg_fun = torch.mean if agg_fun is None else agg_fun
        

    # def logsoftmultimax(self, x):
        # x = x - x.max(1, keepdim=True)[0]
        # denoms = torch.exp(x[:, self.targets_hot_loc[0]]).sum(dim=1)
        # # nums = torch.zeros(x.shape[0], dtype=x.dtype)
        # nums = [torch.exp(x[0, self.targets_hot_loc[0]]).sum()]
        # breakpoint()
        # for k, group in enumerate(self.targets_hot_loc[1:]):
            # nums.append(torch.exp(x[k+1, group]).sum())
            # # nums[k] = torch.exp(x[k, group]).sum()
            # denoms += torch.exp(x[:, group]).sum(dim=1)
        # out = torch.log(torch.tensor(nums)) - torch.log(denoms)
        # return out

    def forward(self, x, target_idx):
        # targets = self.targets[target_idx]
        target_loc = self.targets_hot_loc[target_idx]
        loss = -logsoftmultimaxselect(x, target_loc, self.targets_hot_loc)
        return loss.mean()
        # assert len(x.shape) <= 2
        # targets = self.targets[target_idx]
        # loss = torch.mean(-torch.sum(targets * self.logsoftmultimax(x), dim=1))
        # return loss


if __name__ == '__main__':
    n_classes = 1000
    target_dim = 100
    n_hot = 10
    yh = torch.randn(5, n_classes)
    L = MultihotCrossEntropy(n_classes, target_dim, n_hot)
    y = [800, 454, 3, 55]
    loss = L(yh, y)
=======
>>>>>>> e37bc2104827259356ba1798dd032cd569efeb35

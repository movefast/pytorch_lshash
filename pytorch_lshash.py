import torch
import numpy as np
from torch_utils import *


class PyTorchLSHash(object):
    def __init__(self, hash_size, input_dim):
        self.hash_size = hash_size
        self.uniform_planes = torch.rand(hash_size, input_dim)

    def index(self, input_points):
        planes = self.uniform_planes.t().cuda()
        projections = dot(input_points, planes)
        hashes = projections > 0
        self.hash_tables = dot(
            hashes.float(),
            torch.FloatTensor([2 ** exp for exp in range(self.hash_size, 0, -1)]).unsqueeze(1).cuda(),
        )

    def query(self, s, input_points, number_of_results=100):
        mask_size = len(input_points), 1
        mask = torch.zeros(mask_size).byte().cuda()

        for each in np.unique(self.hash_tables[s].cpu().numpy()):
            new_mask = self.hash_tables.eq(int(each))
            mask += new_mask
        # randomly select number_of_results from existing mask
        m = torch.ones(mask_size) / mask.sum() * number_of_results
        mask = m.squeeze().cuda() * mask.float().cuda()
        mask = torch.bernoulli(mask).byte().unsqueeze(1)

        return torch.masked_select(
            input_points,
            mask.expand_as(input_points),
        ).resize_(mask.sum(), input_points.size()[1])

import math
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np


class StratifiedSampler(DistributedSampler):
    """
    Sampler that restricts data loading to a subset of the dataset,
    and ensures balanced classes in each batch (currently only binary classes)

    See DistributedSampler docs for more details

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset                 : Dataset used for sampling.
        strata (optional)     : Labels to balance among batches
        undersample (optional)  : Fraction of neg/pos samples desired
                                  (e.g. 1.0 for equal;
                                        0.5 for 1/3 neg, 2/3 pos, etc.)
        distributed (optional)  : Stratified DistributedSampler
        num_replicas (optional) : Number of processes participating
                                  in distributed training.
        rank (optional)         : Rank of the current process within
                                  num_replicas.
    """

    def __init__(self,
                 dataset,
                 strata       = None,
                 undersample  = None,
                 distributed  = False,
                 num_replicas = None,
                 rank         = None):

        self.strata      = strata
        self.undersample = undersample
        self.distributed = distributed

        if self.distributed:
            DistributedSampler.__init__(self,
                                        dataset,
                                        num_replicas = num_replicas,
                                        rank         = rank)
        else:
            # TODO need to create iter variables defined in distriburedsampler when strata off
            self.num_replicas = 1
            self.rank = 0
            self.epoch = 0

        if self.strata is not None:
            # NOTE: The return of np.where is a tuple!
            self.pos_stratum = np.where(self.strata == 1)[0]
            self.neg_stratum = np.where(self.strata == 0)[0]

            self.num_pos = int(len(self.pos_stratum))
            self.num_neg = int(len(self.neg_stratum))

            # Per-replica samples
            self.pos_per_replica = int(math.ceil(self.num_pos / self.num_replicas))

            if self.undersample is None:
                self.neg_per_replica = int(math.ceil(self.num_neg / self.num_replicas))
            else:
                self.neg_per_replica = int(self.undersample * self.pos_per_replica)

            self.samples_per_replica = self.pos_per_replica + self.neg_per_replica

            self.pos_total = self.pos_per_replica * self.num_replicas
            self.neg_total = self.neg_per_replica * self.num_replicas

            if self.undersample is not None:
                rng = torch.Generator()
                rng.manual_seed(0)

                neg_indices = torch.randperm(self.num_neg, generator = rng)

                self.neg_indices_init = neg_indices[:self.neg_total]


    def __iter__(self):

        # deterministically shuffle based on epoch
        rng = torch.Generator()
        rng.manual_seed(self.epoch)

        if self.strata is not None:

            pos_indices = torch.randperm(self.num_pos, generator = rng).tolist()

            if self.undersample is not None:
                indices = torch.randperm(len(self.neg_indices_init), generator = rng)
                neg_indices = self.neg_indices_init[indices].tolist()
            else:
                neg_indices = torch.randperm(self.num_neg, generator = rng).tolist()


            # Add extra samples to make it evenly divisible
            pos_indices += pos_indices[ : self.pos_total - len(pos_indices)]
            neg_indices += neg_indices[ : self.neg_total - len(neg_indices)]

            # Subsample data for a rank
            pos_indices = pos_indices[self.rank : self.pos_total : self.num_replicas]
            neg_indices = neg_indices[self.rank : self.neg_total : self.num_replicas]

            # pos/neg to global inds
            pos_indices = self.pos_strata[pos_indices]
            neg_indices = self.neg_strata[neg_indices]

            # interleave
            neg_factor = math.ceil(len(neg_indices) / len(pos_indices))
            indices = []
            for i, j in enumerate(range(0, len(neg_indices), neg_factor)):
                indices.append(pos_indices[i])
                indices.extend(neg_indices[j : j + neg_factor])
        else:
            indices = torch.randperm(len(self.dataset), generator= rng).tolist()

            # Add extra samples to make it evenly divisible
            # NOTE: total_size
            indices += indices[:(self.total_size - len(indices))]
            # Subsample data for a rank
            indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)

    def __len__(self):
        return self.samples_per_replica

    def set_epoch(self, epoch):
        self.epoch = epoch

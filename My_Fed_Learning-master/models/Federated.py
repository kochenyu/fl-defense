import copy
import torch
from torch import nn
import utils.testFunction as test

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedMidden(midden_client_agg):
    agg_midden_value = test.agg_midden_value(midden_client_agg)
    return agg_midden_value
from torch import nn


class AdaptiveModel(nn.Module):
    def __init__(self, model):
        super(AdaptiveModel, self)

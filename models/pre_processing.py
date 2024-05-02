import torch

class logit():
    def __init__(self, alpha = 1e-2):
        self.alpha = alpha

    def transform(self,x, alpha = None):
        assert torch.all(x<=1) and torch.all(x>=0), 'can only transform value between 0 and 1'
        if alpha is None:
            alpha = self.alpha
        return torch.logit(alpha*torch.ones_like(x) + x*(1-2*alpha))

    def inverse_transform(self, x, alpha = None):
        if alpha is None:
            alpha = self.alpha
        return (torch.sigmoid(x)-alpha*torch.ones_like(x))/(1-2*alpha)

    def log_det(self,x, alpha = None ):
        if alpha is None:
            alpha = self.alpha
        return torch.sum(torch.log((1-2*alpha)*(torch.reciprocal(alpha*torch.ones_like(x) + x*(1-2*alpha)) + torch.reciprocal((1-alpha)*torch.ones_like(x) - x*(1-2*alpha)))), dim = -1)

from sklearn import decomposition
import matplotlib.pyplot as plt

class PCA():
    def __init__(self,data, n_components = 'mle',visual = False):
        self.transformer = decomposition.PCA(n_components)
        self.transformer.fit(data)
        if visual:
            values = torch.tensor(self.transformer.explained_variance_ratio_)
            plt.plot(range(len(values)), values)
            plt.plot(range(len(values)), torch.cumsum(values, dim = 0))
            plt.show()

    def transform(self, data):
        return torch.tensor(self.transformer.transform(data)).float()

    def inverse_transform(self, data):
        return torch.tensor(self.transformer.inverse_transform(data)).float()
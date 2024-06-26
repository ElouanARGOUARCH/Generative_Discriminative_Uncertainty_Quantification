import torch
import pyro
from tqdm import tqdm

class discriminative_bayesian_affine_regression:
    def __init__(self,mu_beta=torch.zeros(2), Sigma_beta=torch.eye(2), shape_sigma2=torch.tensor(1.),
                 scale_sigma2=torch.tensor(1.)):
        self.mu_beta = mu_beta
        self.Sigma_beta = Sigma_beta
        self.shape_sigma2 = shape_sigma2
        self.scale_sigma2 = scale_sigma2

    def compute_beta_given_sigma2_D_moments(self,sigma2, DX, DY):
        assert DX.shape == DY.shape, 'Mismatch in number samples'
        temp = torch.cat([DY.unsqueeze(-1), torch.ones(DY.shape[0], 1)], dim=-1)
        Sigma_beta_given_D = torch.inverse(temp.T @ temp / sigma2 + torch.inverse(self.Sigma_beta))
        mu_beta_given_D = Sigma_beta_given_D@(DX @ temp / sigma2 + torch.inverse(self.Sigma_beta)@self.mu_beta)
        return mu_beta_given_D, Sigma_beta_given_D

    def compute_x0_given_y0_beta_sigma2_moments(self, y0, beta,sigma2):
        assert y0.shape[0] == 1, 'Discriminative does not support multiple observations'
        mu_x0_given_y0_beta = beta@torch.cat([y0,torch.ones_like(y0)], dim = -1)
        sigma2_x0_given_y0_beta = sigma2
        return mu_x0_given_y0_beta, sigma2_x0_given_y0_beta

    def compute_sigma2_given_beta_D_parameters(self, beta, DX, DY):
        assert DX.shape[0] == DY.shape[0], 'Mismatch in number samples'
        temp = torch.cat([DY.unsqueeze(-1), torch.ones(DY.shape[0], 1)], dim=-1)
        shape_N = self.shape_sigma2 + DX.shape[0] / 2
        scale_N = self.scale_sigma2 + torch.sum(torch.square(DX - temp @ beta)) / 2
        estimated_sigma2 = pyro.distributions.InverseGamma(shape_N, scale_N).sample()
        return estimated_sigma2

    def sample_x0_given_y0_D_Y_gibbs(self, y0, DX, DY,Y=torch.tensor([]),number_steps=100, verbose=False):
        assert DX.shape[0] == DY.shape[0], 'mismatch in dataset numbers'
        DYplus = torch.cat([DY, y0, torch.flatten(Y)], dim=0)
        current_sigma2 = pyro.distributions.InverseGamma(self.shape_sigma2, self.scale_sigma2).sample()
        mean_beta_given_sigma2_D, Sigma_beta_given_sigma2_D = self.compute_beta_given_sigma2_D_moments(current_sigma2, DX, DY)
        current_beta = torch.distributions.MultivariateNormal(mean_beta_given_sigma2_D, Sigma_beta_given_sigma2_D).sample()
        list_x0_gibbs = []
        list_beta_gibbs = []
        list_sigma2_gibbs = []
        if verbose:
            pbar = tqdm(range(number_steps))
        else:
            pbar = range(number_steps)
        for _ in pbar:
            mean_x0_given_y0_beta_sigma2, sigma2_x0_given_y0_beta_sigma2 = self.compute_x0_given_y0_beta_sigma2_moments(
                y0, current_beta, current_sigma2)
            current_x0 = (mean_x0_given_y0_beta_sigma2 + torch.sqrt(sigma2_x0_given_y0_beta_sigma2)*torch.randn(1)).squeeze(-1)
            current_labels = []
            for yj in Y:
                mu_xj_given_yj_beta_sigma2,sigma2_xj_given_yj_beta_sigma2 = self.compute_x0_given_y0_beta_sigma2_moments(yj, current_beta, current_sigma2)
                current_label = (mu_xj_given_yj_beta_sigma2 + torch.sqrt(sigma2_xj_given_yj_beta_sigma2)*torch.randn(1)).squeeze(-1)
                current_labels.append(current_label.repeat(yj.shape[0]))
            if torch.flatten(Y).shape[0] > 0:
                DXplus = torch.cat([DX, current_x0.repeat(y0.shape[0]), torch.cat(current_labels)], dim=0)
            else:
                DXplus = torch.cat([DX, current_x0.repeat(y0.shape[0])])
            mean_beta_given_Dplus, Sigma_beta_given_Dplus = self.compute_beta_given_sigma2_D_moments(current_sigma2,
                                                                                                     DXplus, DYplus)
            current_beta = (mean_beta_given_Dplus + torch.linalg.cholesky(Sigma_beta_given_Dplus)@torch.randn(2)).squeeze(-1)
            current_sigma2 = self.compute_sigma2_given_beta_D_parameters(current_beta, DXplus, DYplus)
            list_x0_gibbs.append(current_x0)
            list_beta_gibbs.append(current_beta)
            list_sigma2_gibbs.append(current_sigma2)
        return torch.stack(list_x0_gibbs), torch.stack(list_beta_gibbs), torch.stack(list_sigma2_gibbs)
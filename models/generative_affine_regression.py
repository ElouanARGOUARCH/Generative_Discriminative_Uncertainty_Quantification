import torch
from tqdm import tqdm
import pyro

class generative_bayesian_affine_regression_known_variance:
    def __init__(self, sigma2_simulateur, mu_X=torch.tensor(0.), sigma2_X=torch.tensor(1.),
                 mu_beta=torch.zeros(2), Sigma_beta=torch.eye(2)):
        self.sigma2_simulateur = sigma2_simulateur
        self.mu_X = mu_X
        self.sigma2_X = sigma2_X
        self.mu_beta = mu_beta
        self.Sigma_beta = Sigma_beta

    def compute_x0_given_y0_beta_moments(self, y0, beta):
        sigma_x0_given_y0_beta = 1 / (
                    1 / self.sigma2_X + (y0.shape[0] * torch.square(beta[0])) / self.sigma2_simulateur)
        mu_x0_given_y0_beta = sigma_x0_given_y0_beta * (
                    self.mu_X / self.sigma2_X + beta[0] * torch.sum(y0 - beta[1]) / self.sigma2_simulateur)
        return mu_x0_given_y0_beta, sigma_x0_given_y0_beta

    def compute_beta_given_D_moments(self, DX, DY):
        assert DX.shape[0] == DY.shape[0], 'Mismatch in number samples'
        if DX.shape[0] >= 1:
            temp = torch.cat([DX.unsqueeze(-1), torch.ones(DX.shape[0], 1)], dim=-1)
            Sigma_beta_given_D = torch.inverse(
                temp.T @ temp / self.sigma2_simulateur + torch.inverse(self.Sigma_beta))
            mu_beta_given_D = Sigma_beta_given_D @ (
                        DY @ temp / self.sigma2_simulateur + torch.inverse(self.Sigma_beta)@self.mu_beta)
        else:
            mu_beta_given_D, Sigma_phi_given_D = self.mu_phi, self.sigma_phi
        return mu_beta_given_D, Sigma_beta_given_D

    def x0_beta_given_y0_D_log_joint(self, x0,beta,y0,DX,DY):
        assert x0.shape[0] == beta.shape[0],'Mismatch in number of samples'
        x0_log_prior = torch.distributions.Normal(self.mu_X, torch.sqrt(self.sigma2_X)).log_prob(x0)

        mean_beta_given_D, Sigma_beta_given_D = self.compute_beta_given_D_moments(DX, DY)
        beta_given_D_log_posterior = torch.distributions.MultivariateNormal(mean_beta_given_D,
                                                                 Sigma_beta_given_D).log_prob(beta)

        augmented_x0 = torch.cat([x0.unsqueeze(-1), torch.ones(x0.shape[0], 1)], dim=-1)
        temp = torch.bmm(beta.unsqueeze(-2), augmented_x0.unsqueeze(-1)).squeeze(-1)
        temp = temp.repeat(1, y0.shape[0])
        cov_matrix = self.sigma2_simulateur * torch.eye(y0.shape[0]).unsqueeze(0).repeat(beta.shape[0], 1, 1)
        y0_given_x0_beta_log_likelihood = torch.distributions.MultivariateNormal(temp, cov_matrix).log_prob(y0) if \
        y0.shape[0] >= 1 else torch.zeros(beta.shape[0])

        return beta_given_D_log_posterior + x0_log_prior + y0_given_x0_beta_log_likelihood

    def x0_given_y0_D_marginal_log_posterior(self, x0, y0, DX, DY):
        mean_beta_given_D, Sigma_beta_given_D = self.compute_beta_given_D_moments(DX, DY)
        gamma = torch.cat([x0.unsqueeze(-1), torch.ones_like(x0).unsqueeze(-1)], dim=-1).unsqueeze(-2).repeat(
            1, y0.shape[0], 1)
        x0_log_prior = torch.distributions.Normal(self.mu_X, self.sigma2_X).log_prob(x0)
        yO_given_x0_D_marginal_log_likelihood = torch.distributions.MultivariateNormal(gamma @ mean_beta_given_D,
                                                            gamma @ Sigma_beta_given_D @ gamma.mT + self.sigma2_simulateur * torch.eye(
                                                                y0.shape[0])).log_prob(
            y0.unsqueeze(0).repeat(x0.shape[0], 1))
        return x0_log_prior + yO_given_x0_D_marginal_log_likelihood

    def sample_x0_given_y0_D_Y_gibbs(self, y0,DX, DY,Y = torch.tensor([]),prior_means = torch.tensor([]),prior_sigma2s = torch.tensor([]),number_steps = 100,verbose = False):
        assert Y.shape[0]==len(prior_means)==len(prior_sigma2s),'mismatch in number of unlabeled samples and specified priors'
        assert DX.shape[0]==DY.shape[0],'mismatch in dataset numbers'
        DYplus = torch.cat([DY, y0,torch.flatten(Y)], dim=0)
        current_x0 = torch.distributions.Normal(self.mu_X, torch.sqrt(self.sigma2_X)).sample()
        current_labels = []
        for y, prior_mean, prior_sigma2 in zip(Y, prior_means, prior_sigma2s):
            current_label = torch.distributions.Normal(prior_mean,
                                                       torch.sqrt(prior_sigma2)).sample()
            current_labels.append(current_label.repeat(y.shape[0]))
        if torch.flatten(Y).shape == 0:
            DXplus = torch.cat([DX, current_x0.repeat(y0.shape[0]), torch.cat(current_labels)], dim=0)
        else:
            DXplus = torch.cat([DX, current_x0.repeat(y0.shape[0])])
        list_x0_gibbs = []
        list_beta_gibbs = []
        if verbose:
            pbar = tqdm(range(number_steps))
        else:
            pbar = range(number_steps)
        for _ in pbar:
            mean_beta_given_Dplus, Sigma_beta_given_Dplus = self.compute_beta_given_D_moments(DXplus, DYplus)
            current_beta = torch.distributions.MultivariateNormal(mean_beta_given_Dplus,Sigma_beta_given_Dplus).sample()
            mu_x0_given_y0_beta, sigma2_x0_given_y0_beta = self.compute_x0_given_y0_beta_moments(y0, current_beta)
            current_x0 = torch.distributions.Normal(mu_x0_given_y0_beta, torch.sqrt(sigma2_x0_given_y0_beta)).sample()
            current_labels = []
            for y, prior_mean, prior_sigma2 in zip(Y, prior_means, prior_sigma2s):
                sigma2_xj_given_yj_beta = 1 / (
                        1 / prior_sigma2 + (y.shape[0] * torch.square(current_beta[0])) / self.sigma2_simulateur)
                mu_xj_given_yj_beta = sigma2_xj_given_yj_beta * (
                        prior_mean / prior_sigma2 + current_beta[0] * torch.sum(y - current_beta[1]) / self.sigma2_simulateur)
                current_label = torch.distributions.Normal(mu_xj_given_yj_beta,
                                                           torch.sqrt(sigma2_xj_given_yj_beta)).sample()
                current_labels.append(current_label.repeat(y.shape[0]))
            if torch.flatten(Y).shape==0:
                DXplus = torch.cat([DX, current_x0.repeat(y0.shape[0]), torch.cat(current_labels)], dim=0)
            else:
                DXplus = torch.cat([DX, current_x0.repeat(y0.shape[0])])
            list_x0_gibbs.append(current_x0)
            list_beta_gibbs.append(current_beta)
        return torch.stack(list_x0_gibbs), torch.stack(list_beta_gibbs)

class generative_bayesian_affine_regression:
    def __init__(self, mu_X=torch.tensor(0.), sigma2_X=torch.tensor(1.),
                 mu_beta=torch.zeros(2), Sigma_beta=torch.eye(2), shape_sigma2=torch.tensor(1.),
                 scale_sigma2=torch.tensor(1.)):
        self.mu_X = mu_X
        self.sigma2_X = sigma2_X
        self.mu_beta = mu_beta
        self.Sigma_beta = Sigma_beta
        self.shape_sigma2 = shape_sigma2
        self.scale_sigma2 = scale_sigma2

    def compute_x0_given_y0_beta_sigma2_moments(self, y0, beta, sigma2):
        sigma_x0_given_y0_beta = 1 / (
                1 / self.sigma2_X + (y0.shape[0] * torch.square(beta[0])) / sigma2)
        mu_x0_given_y0_beta = sigma_x0_given_y0_beta * (
                self.mu_X / self.sigma2_X + beta[0] * torch.sum(y0 - beta[1]) / sigma2)
        return mu_x0_given_y0_beta, sigma_x0_given_y0_beta

    def compute_beta_given_sigma2_D_moments(self, sigma2, DX, DY):
        assert DX.shape[0] == DY.shape[0], 'Mismatch in number samples'
        temp = torch.cat([DX.unsqueeze(-1), torch.ones(DX.shape[0], 1)], dim=-1)
        Sigma_beta_given_D = torch.inverse(
            temp.T @ temp / sigma2 + torch.inverse(self.Sigma_beta))
        mu_beta_given_D = Sigma_beta_given_D @ (
                DY @ temp / sigma2 + torch.inverse(self.Sigma_beta) @ self.mu_beta)
        return mu_beta_given_D, Sigma_beta_given_D

    def compute_sigma2_given_beta_D_parameters(self, beta, DX, DY):
        assert DX.shape[0] == DY.shape[0], 'Mismatch in number samples'
        temp = torch.cat([DX.unsqueeze(-1), torch.ones(DX.shape[0], 1)], dim=-1)
        x_N = DY - temp @ beta
        N = x_N.shape[0]
        shape_N = self.shape_sigma2 + N / 2
        scale_N = self.scale_sigma2 + torch.sum(torch.square(x_N)) / 2
        estimated_sigma2 = pyro.distributions.InverseGamma(shape_N, scale_N).sample()
        return estimated_sigma2

    def sample_x0_given_y0_D_Y_gibbs(self, y0, DX, DY, Y=torch.tensor([]), prior_means=torch.tensor([]),
                                     prior_sigma2s=torch.tensor([]), number_steps=100, verbose=False):
        assert Y.shape[0] == len(prior_means) == len(prior_sigma2s), 'mismatch in number of unlabeled samples and specified priors'
        assert DX.shape[0] == DY.shape[0], 'mismatch in dataset numbers'
        DYplus = torch.cat([DY, y0, torch.flatten(Y)], dim=0)
        current_sigma2 = pyro.distributions.InverseGamma(self.shape_sigma2, self.scale_sigma2).sample()
        mean_beta_given_D, Sigma_beta_given_D = self.compute_beta_given_sigma2_D_moments(current_sigma2, DX, DY)
        current_beta = torch.distributions.MultivariateNormal(mean_beta_given_D, Sigma_beta_given_D).sample()
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
            for y, prior_mean, prior_sigma2 in zip(Y, prior_means, prior_sigma2s):
                sigma2_xj_given_yj_beta_sigma2 = 1 / (
                        1 / prior_sigma2 + (y.shape[0] * torch.square(current_beta[0])) / current_sigma2)
                mu_xj_given_yj_beta_sigma2 = sigma2_xj_given_yj_beta_sigma2 * (
                        prior_mean / prior_sigma2 + current_beta[0] * torch.sum(
                    y - current_beta[1]) / current_sigma2)
                current_label = (mu_xj_given_yj_beta_sigma2 + torch.sqrt(sigma2_xj_given_yj_beta_sigma2)*torch.randn(1)).squeeze(-1)
                current_labels.append(current_label.repeat(y.shape[0]))
            if torch.flatten(Y).shape[0] > 0:
                DXplus = torch.cat([DX, current_x0.repeat(y0.shape[0]), torch.cat(current_labels)], dim=0)
            else:
                DXplus = torch.cat([DX, current_x0.repeat(y0.shape[0])])
            mean_beta_given_Dplus, Sigma_beta_given_Dplus = self.compute_beta_given_sigma2_D_moments(current_sigma2,DXplus, DYplus)
            current_beta = (mean_beta_given_Dplus + torch.linalg.cholesky(Sigma_beta_given_Dplus)@torch.randn(2)).squeeze(-1)
            current_sigma2 = self.compute_sigma2_given_beta_D_parameters(current_beta, DXplus, DYplus)
            list_x0_gibbs.append(current_x0)
            list_beta_gibbs.append(current_beta)
            list_sigma2_gibbs.append(current_sigma2)
        return torch.stack(list_x0_gibbs), torch.stack(list_beta_gibbs), torch.stack(list_sigma2_gibbs)
import torch
from tqdm import tqdm

class Classifier(torch.nn.Module):
    def __init__(self, sample_dim, C, hidden_dimensions=[]):
        super().__init__()
        self.sample_dim = sample_dim
        self.C = C
        self.network_dimensions = [self.sample_dim] + hidden_dimensions + [self.C]
        network = []
        for h0, h1 in zip(self.network_dimensions, self.network_dimensions[1:]):
            network.extend([torch.nn.Linear(h0, h1), torch.nn.Tanh(), ])
        self.f = torch.nn.Sequential(*network)

    def compute_number_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def log_prob(self, samples):
        temp = self.f.forward(samples)
        return temp - torch.logsumexp(temp, dim=-1, keepdim=True)

    def loss(self, samples, labels):
        return -torch.mean(self.log_prob(samples) * labels)

    def train(self, epochs, batch_size,train_samples, train_labels,list_test_samples = [], list_test_labels = [],verbose = False, recording_frequency = 1, lr=5e-4, weight_decay=5e-5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters())
        dataset = torch.utils.data.TensorDataset(train_samples, train_labels)
        if verbose:
            train_loss_trace = []
            list_test_loss_trace = [[] for i in range(len(list_test_samples))]
        pbar = tqdm(range(epochs))
        for __ in pbar:
            self.to(device)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for _, batch in enumerate(dataloader):
                optimizer.zero_grad()
                loss = self.loss(batch[0].to(device), batch[1].to(device))
                loss.backward()
                optimizer.step()
            if __ % recording_frequency == 0 and verbose:
                with torch.no_grad():
                    self.to(torch.device('cpu'))
                    train_loss = self.loss(train_samples, train_labels).item()
                    train_loss_trace.append(train_loss)
                    postfix_str = 'device = ' + str(
                        device) + '; train_loss = ' + str(round(train_loss, 4))
                    for i in range(len(list_test_samples)):
                        test_loss = self.loss(list_test_samples[i], list_test_labels[i]).item()
                        list_test_loss_trace[i].append(test_loss)
                        postfix_str += '; test_loss_'+ str(i) +' = ' + str(round(test_loss, 4))
                    pbar.set_postfix_str(postfix_str)
        self.to(torch.device('cpu'))
        if verbose:
            return train_loss_trace, list_test_loss_trace

class LogisticClassifier:
    def __init__(self, samples, labels, mu_beta=None, Sigma_beta=None):
        self.samples = samples
        self.labels = labels
        self.sample_dim = samples.shape[-1]
        self.C = labels.shape[-1]
        if mu_beta is not None:
            self.mu_beta = mu_beta
        else:
            self.mu_beta = torch.zeros((self.sample_dim + 1) * self.C)
        if Sigma_beta is not None:
            self.Sigma_beta = Sigma_beta
        else:
            self.Sigma_beta = 5 * torch.eye((self.sample_dim + 1) * self.C)

    def sample_beta_from_prior(self, num_samples):
        return torch.distributions.MultivariateNormal(self.mu_beta, self.Sigma_beta).sample(num_samples)

    def log_prob(self, samples, betas):
        log_prob = -torch.cat([samples, torch.ones(samples.shape[0], 1)], dim=-1).unsqueeze(0).repeat(betas.shape[0], 1,
                                                                                                      1) @ betas.reshape(
            betas.shape[0], self.sample_dim + 1, self.C)
        return log_prob - torch.logsumexp(log_prob, dim=-1, keepdim=True)

    def log_posterior_prob(self, betas):
        return torch.sum(self.log_prob(self.samples, betas) * self.labels.unsqueeze(0).repeat(betas.shape[0], 1, 1),
                         dim=[-2, -1]) + torch.distributions.MultivariateNormal(self.mu_beta, self.Sigma_beta).log_prob(
            betas)

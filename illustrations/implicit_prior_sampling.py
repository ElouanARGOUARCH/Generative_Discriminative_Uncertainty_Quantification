from models import *
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec

scale_sigma2 = torch.tensor(.5)
shape_sigma2 = torch.tensor(.5)
sigma20 = torch.tensor(1)/20

mu_beta = torch.zeros(2)
Sigma_beta = torch.eye(2)
alpha = torch.distributions.MultivariateNormal(mu_beta, Sigma_beta).sample()
alpha = torch.tensor([.5, 1.])

DGP = lambda x: alpha[0] * x + alpha[1] + torch.randn(x.shape[0]) * torch.sqrt(sigma20)

mu_DX = torch.tensor(1.5)
sigma2_DX = torch.tensor(1.) / 4
prior_dataset = torch.distributions.Normal(mu_DX, torch.sqrt(sigma2_DX))
n_D = 250
DX = prior_dataset.sample([n_D])
DY = DGP(DX)

mu_X0 = torch.tensor(4.5)
sigma2_X0 = torch.tensor(1) / 4
inference_prior = torch.distributions.Normal(mu_X0, torch.sqrt(sigma2_X0))

mu_Y0 = alpha[0] * mu_X0 + alpha[1]
sigma2_Y0 = sigma2_X0 * alpha[0] ** 2 + sigma20
y_marginal = torch.distributions.Normal(mu_Y0, torch.sqrt(sigma2_Y0))

model = discriminative_bayesian_affine_regression()
number_MC_samples = 1000
list_x = []
for i in range(number_MC_samples):
    x0 = inference_prior.sample()
    y0 = DGP(x0.unsqueeze(0))
    x, _, _ = model.sample_x0_given_y0_D_Y_gibbs(y0, DX, DY)
    list_x.append(x[-10:])

fig = plt.figure(figsize=(20,10))
gs = GridSpec(2,2, figure = fig, width_ratios=[3,2], height_ratios=[1,1])
ax = fig.add_subplot(gs[0,0])
tt = torch.linspace(0, 6, 300)
ax.set_xlim(0, 6)
ax.set_ylim(0, 5)
tt_y = torch.linspace(0, 5, 300)

list_mean_ppd = []
list_sigma_ppd = []
for y in tt_y:
    x, beta, sigma2 = model.sample_x0_given_y0_D_Y_gibbs(y.unsqueeze(0), DX, DY, number_steps=100)
    mean, var = model.compute_x0_given_y0_beta_sigma2_moments(y.unsqueeze(0), beta, sigma2)
    list_mean_ppd.append(torch.mean(mean).unsqueeze(0))
    list_sigma_ppd.append(
        torch.sqrt(torch.mean(var) + torch.mean(torch.square(mean)) - torch.square(torch.mean(mean))).unsqueeze(0))
means_ppd = torch.cat(list_mean_ppd)
sigma_ppd = torch.cat(list_sigma_ppd)

ax.plot(tt.numpy(), (alpha[0] * tt + alpha[1]).numpy(), linestyle='--', label=r'$y = \alpha_1x+ \alpha_0$', color='C0')
ax.fill_between(tt.numpy(), (alpha[0] * tt + alpha[1]).numpy() - 3 * torch.sqrt(sigma20).numpy(),
                (alpha[0] * tt + alpha[1]).numpy() + 3 * torch.sqrt(sigma20).numpy(), linestyle='--', color='C0',
                alpha=0.25, label=r'$\sigma_{Y|X}$')
ax.scatter(DX.numpy(), DY.numpy(), alpha=.5, label=r'$\mathcal{D}=\{(x_i,y_i)\}$', color='C1')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(tt, torch.exp(inference_prior.log_prob(tt.unsqueeze(-1))), color='green', label=r'prior $\pi_{X_0}$')
ax.plot(torch.exp(y_marginal.log_prob(tt_y)), tt_y, linestyle='--', color='green', label=r'$\mathcal{P}_{Y_0}$')
ax.plot(tt, torch.exp(prior_dataset.log_prob(tt)), linestyle='--', label=r'$\mathcal{P}_{X}^{\mathcal{D}}$', color='C1')
ax.legend(ncol=1, fontsize=20, loc='upper right')

#plt.subplots_adjust(wspace=0.05, hspace=0)
#plt.show()
#fig = plt.figure(figsize=(20, 6))
ax = fig.add_subplot(gs[1,0])
ax.set_xlim(0, 6)
ax.set_ylim(0, 5)
ax.scatter(DX.numpy(), DY.numpy(), alpha=.5, label=r'$\mathcal{D}=\{(x_i,y_i)\}$', color='C1')
plt.plot(means_ppd, tt_y, color='pink')
plt.fill_betweenx(tt_y, means_ppd - 3 * sigma_ppd, means_ppd + 3 * sigma_ppd, color='pink', alpha=.3,
                  label=r'ppd $p(x_0|y_0,\mathcal{D})$')
ax.plot(torch.exp(y_marginal.log_prob(tt_y)), tt_y, linestyle='--', color='green', label=r'$\mathcal{P}_{Y_0}$')
ax.hist(torch.cat(list_x).numpy(), bins=25, density=True, histtype='step', label=r'$p(x_0|\mathcal{D})$ dis.',
        color='C9')
ax.legend(ncol=1, fontsize=20, loc='upper right')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

#plt.show()
#fig = plt.figure(figsize=(12, 12))
#ax = fig.add_subplot(111)

ax = fig.add_subplot(gs[:,1])
ax.plot(tt, torch.exp(prior_dataset.log_prob(tt)), linestyle='--', label=r'$\mathcal{P}_{X}^{\mathcal{D}}$', color='C1')
ax.hist(torch.cat(list_x).numpy(), bins=60, density=True, histtype='step', label=r'$p(x_0|\mathcal{D})$ dis.',
        color='C9')
ax.plot(tt, torch.exp(inference_prior.log_prob(tt.unsqueeze(-1))), color='green', label=r'prior $\pi_{X_0}$')

ax.legend(ncol=1, fontsize=16.6, loc='upper left')
ax.set_xlim(0, 6)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.tight_layout()
fig.savefig('figure.png', dpi = fig.dpi)
plt.show()
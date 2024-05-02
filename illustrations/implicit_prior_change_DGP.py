from models import *
from matplotlib.ticker import MaxNLocator
list_samples = []
sigma20s = torch.tensor([10, 0.1, .025, 0.001])
for sigma20 in tqdm(sigma20s):
    scale_sigma2 = torch.tensor(.5)
    shape_sigma2 = torch.tensor(.5)

    mu_beta = torch.zeros(2)
    Sigma_beta = torch.eye(2)
    alpha = torch.distributions.MultivariateNormal(mu_beta, Sigma_beta).sample()
    alpha = torch.tensor([.5, 1.])

    DGP = lambda x: alpha[0] * x + alpha[1] + torch.randn(x.shape[0]) * torch.sqrt(sigma20)

    mu_DX = torch.tensor(1.)
    sigma2_DX = torch.tensor(1.) / 4
    prior_dataset = torch.distributions.Normal(mu_DX, torch.sqrt(sigma2_DX))
    n_D = 250
    DX = prior_dataset.sample([n_D])
    DY = DGP(DX)

    mu_X0 = torch.tensor(5.)
    sigma2_X0 = torch.tensor(1) / 4
    inference_prior = torch.distributions.Normal(mu_X0, torch.sqrt(sigma2_X0))

    mu_Y0 = alpha[0] * mu_X0 + alpha[1]
    sigma2_Y0 = sigma2_X0 * alpha[0] ** 2 + sigma20
    y_marginal = torch.distributions.Normal(mu_Y0, torch.sqrt(sigma2_Y0))

    model = discriminative_bayesian_affine_regression()
    number_MC_samples = 20000
    list_x = []
    for i in range(number_MC_samples):
        x0 = inference_prior.sample()
        y0 = DGP(x0.unsqueeze(0))
        x, _, _ = model.sample_x0_given_y0_D_Y_gibbs(y0, DX, DY)
        list_x.append(x[-1:])
    list_samples.append(torch.cat(list_x))

fig = plt.figure(figsize=(24, 6))
for i, list_x in enumerate(list_samples):
    ax = fig.add_subplot(1, 4, i + 1)
    tt = torch.linspace(-1, 7, 300)
    ax.plot(tt, torch.exp(prior_dataset.log_prob(tt)), linestyle='--', label=r'$\mathcal{P}_{X}^{\mathcal{D}}$',
            color='C1')
    ax.hist(list_x.numpy(), bins=80, density=True, histtype='step', label=r'$p(x_0|\mathcal{D})$ dis.', color='C9')
    ax.plot(tt, torch.exp(inference_prior.log_prob(tt.unsqueeze(-1))), color='green', label=r'prior $\pi_{X_0}$')

    ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
    ax.set_xlim(-1, 7)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.text(0, .875, r'$\sigma_{Y|X}^2 =$' + str(round(sigma20s[i].item(), 3)), size=16.6)
plt.subplots_adjust(wspace=0.05, hspace=0)
plt.legend(ncol=1, fontsize=16.6, loc='upper right')
plt.show()
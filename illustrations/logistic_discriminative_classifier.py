from markov_chain_monte_carlo import *
from models import *

inference_prior = torch.tensor([3,2,1])/6
dataset_prior = torch.tensor([1,2,3])/6
r = np.arange(3)
total_numbers = 2000
width = 0.25

locations = lambda m: m*torch.tensor([[torch.cos(0*4*torch.acos(torch.zeros(1))/3),torch.sin(0*4*torch.acos(torch.zeros(1))/3)]
                            ,[torch.cos(1*4*torch.acos(torch.zeros(1))/3),torch.sin(1*4*torch.acos(torch.zeros(1))/3)]
                            ,[torch.cos(2*4*torch.acos(torch.zeros(1))/3),torch.sin(2*4*torch.acos(torch.zeros(1))/3)]])

list_m = [0.10,.7,1.4,2.,5.]
list_lr = [2.4e-4,3.4e-4,4.8e-4,5.4e-4,2.8e-4]

def sample_DGP(number_samples,locations, prior):
    x = torch.distributions.Categorical(prior)
    mvn = torch.distributions.MultivariateNormal(locations, torch.eye(2).unsqueeze(0).repeat(locations.shape[0],1,1))
    labels  = torch.nn.functional.one_hot(x.sample(number_samples), num_classes = prior.shape[0])
    obs = torch.sum(mvn.sample(number_samples)*labels.unsqueeze(-1), dim = 1)
    return obs, labels

def plot_2d_contour(f,range = [[-10,10],[-10,10]], bins = [50,50],levels = 4, alpha = 0.7,show = True,color = None):
    with torch.no_grad():
        tt_x = torch.linspace(range[0][0], range[0][1], bins[0])
        tt_y = torch.linspace(range[1][0],range[1][1], bins[1])
        mesh = torch.cartesian_prod(tt_x, tt_y)
        with torch.no_grad():
            plot=plt.contour(tt_x,tt_y,f(mesh).numpy().reshape(bins[0],bins[1]).T,levels = levels, colors = color, alpha = alpha)
    if show:
        plt.show()
    return plot
list_list_prob = []
for m,lr in zip(list_m, list_lr):
    samples_obs, labels_obs = sample_DGP([5000],locations(m),inference_prior)
    plt.figure(figsize = (6,6))
    plt.xlim([-6,6])
    plt.ylim([-6,6])
    list_handles = []
    for i in r:
        distribution = torch.distributions.MultivariateNormal(locations(m)[i], torch.eye(2))
        contour = plot_2d_contour(lambda x: torch.exp(distribution.log_prob(x)),bins = [200,200], range =[[-5,5],[-5,5]],levels = 5, show = False, color = 'C'+str(i+4))
        handles, labels = contour.legend_elements()
        list_handles.append(handles[0])
    plt.legend(list_handles, [r'$\mathcal{P}_{Y|X=1}$',r'$\mathcal{P}_{Y|X=2}$',r'$\mathcal{P}_{Y|X=3}$',r'$\mathcal{P}_{Y|X=4}$'], ncol = 1, fontsize = 12, loc = 'upper right')
    plt.show()

    index = torch.distributions.Categorical(dataset_prior/torch.sum(dataset_prior)).sample([total_numbers])
    number_samples= torch.sum(torch.nn.functional.one_hot(index), dim = 0)

    fig = plt.figure(figsize = (11,5))
    ax1 = fig.add_subplot(121)
    ax1.bar(r,(dataset_prior/torch.sum(dataset_prior)).numpy(),width = width, color = 'C1',edgecolor = 'black', label =r'$\mathcal{P}_X^{\mathcal{D}}$')
    ax1.set_xticks(r,[str(i) for i in r])
    ax1.set_xlabel("X")
    ax1.set_ylabel("Probabilities")
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.set_xlim([-6,6])
    ax2.set_ylim([-6,6])
    samples_0 = torch.randn(number_samples[0],2)+locations(m)[0]
    samples_1 = torch.randn(number_samples[1],2)+locations(m)[1]
    samples_2 = torch.randn(number_samples[2],2)+locations(m)[2]
    ax2.scatter(samples_0[:,0].numpy(),samples_0[:,1].numpy(), alpha = .6, label ="class 1", color='C' +str(4))
    ax2.scatter(samples_1[:,0].numpy(),samples_1[:,1].numpy(), alpha = .6, label ="class 2", color='C' +str(5))
    ax2.scatter(samples_2[:,0].numpy(),samples_2[:,1].numpy(), alpha = .6, label ="class 3", color='C' +str(6))
    ax2.legend(ncol = 1, fontsize = 12,loc = 'upper right')

    samples = torch.cat([samples_0, samples_1, samples_2])
    labels = torch.cat([torch.zeros(number_samples[0]), torch.ones(number_samples[1]), 2*torch.ones(number_samples[2])]).long()
    plt.show()

    model = LogisticClassifier(samples, torch.nn.functional.one_hot(labels))
    sampler = Langevin(lambda beta: model.log_posterior_prob(beta),model.sample_beta_from_prior([1]).shape[-1],None,1000)
    samples_beta_post = sampler.sample_MALA(100,tau = lr,verbose=True)
    with torch.no_grad():
        list_prob = torch.mean(torch.exp(model.log_prob(samples_obs,samples_beta_post)), dim =[0,1])

    plt.figure()
    ax = plt.subplot(111)
    ax.bar(r+2*width,(dataset_prior/torch.sum(dataset_prior)).numpy(),width = width, color = 'C1',edgecolor = 'black', label =r'$\mathcal{P}_X^{\mathcal{D}}$')
    ax.set_xticks(r+3*width,[str(i) for i in r])
    ax.set_xlabel("X")
    ax.set_ylabel("Probability")
    ax.bar(r+3*width, list_prob, color = 'C9',
            width = width, edgecolor = 'black',
            label=r'$p(x_0|\mathcal{D})$')
    list_list_prob.append(list_prob)
    ax.bar(r+4*width, inference_prior, color = 'green',
            width = width, edgecolor = 'black',
            label=r'$\Pi_{X_0}$')


    ax.legend()
    plt.xlabel("X")
    plt.ylabel("Probability")

    plt.legend()

    plt.show()
fig = plt.figure(figsize = (30,6))
for i,list_prob in enumerate(list_list_prob):
    ax = plt.subplot(1,len(list_m),i+1)
    ax.bar(r+2*width,(dataset_prior/torch.sum(dataset_prior)).numpy(),width = width, color = 'C1',edgecolor = 'black', label =r'$\mathcal{P}_X^{\mathcal{D}}$')
    ax.set_xticks(r+3*width,['1','2','3'])
    ax.set_xlabel("X")
    ax.set_ylabel("Probabilities")
    ax.bar(r+3*width, list_prob, color = 'C9',
            width = width, edgecolor = 'black',
            label=r'$p(x_0|\mathcal{D})$')
    ax.bar(r+4*width, inference_prior, color = 'green',
            width = width, edgecolor = 'black',
            label=r'$\Pi_{X_0}$')
    ax.set_xlabel("X")
    ax.set_ylabel("Probability")
    ax.text(1.4,.45, r'$r_{Y|X} =$' +str(round(list_m[i],3)), size = 16.6)

plt.subplots_adjust(wspace=0.15, hspace=0)
plt.legend(ncol = 1, fontsize = 13,loc = 'upper right')
plt.show()
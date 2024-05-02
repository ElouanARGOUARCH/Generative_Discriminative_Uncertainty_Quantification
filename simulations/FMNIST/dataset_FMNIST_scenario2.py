from models import *

logit_transform = logit(alpha=1e-6)
samples, labels = get_FashionMNIST_dataset(one_hot=True)
samples, randperm = shuffle(logit_transform.transform(samples))
labels, _ = shuffle(labels, randperm)
labels = labels.float()
pca_transform = PCA(samples, n_components=100)
samples = pca_transform.transform(samples)
num_samples = torch.sum(labels, dim=0)

r = range(0, 10)
train_prior_probs = torch.tensor([(i + 1) for i in r]) * num_samples
train_prior_probs = train_prior_probs / torch.sum(train_prior_probs)
test_prior_probs = torch.tensor([10 - i for i in r]) * num_samples
test_prior_probs = test_prior_probs / torch.sum(test_prior_probs)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.bar(r, train_prior_probs, color = 'C1', alpha = .4, label = r'$\mathcal{P}_X^{\mathcal{D}}$')
ax.bar(r, test_prior_probs, color = 'green', alpha = .4, label = r'$\Pi_{X_0}$')
plt.legend(fontsize = 25)
plt.show()

train_samples, train_labels = [], []
test_samples, test_labels = [], []
for label in range(labels.shape[-1]):
    current_samples = samples[labels[:, label] == 1]
    current_labels = labels[labels[:, label] == 1]
    for_train = current_samples.shape[0] * train_prior_probs[label] / (
                train_prior_probs[label] + test_prior_probs[label])
    train_samples.append(current_samples[:int(for_train)])
    test_samples.append(current_samples[int(for_train):])
    train_labels.append(current_labels[:int(for_train)])
    test_labels.append(current_labels[int(for_train):])
train_samples, train_labels = torch.cat(train_samples), torch.cat(train_labels)
test_samples, test_labels = torch.cat(test_samples), torch.cat(test_labels)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(r, torch.sum(train_labels, dim = 0), color = 'C1', alpha = .4)
ax.bar(r, torch.sum(test_labels, dim = 0), color = 'green', alpha = .4)
plt.show()
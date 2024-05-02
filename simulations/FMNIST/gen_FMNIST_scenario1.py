from models import *

number_runs = 10

for run in range(number_runs):
    logit_transform = logit(alpha = 1e-6)
    samples, labels = get_FashionMNIST_dataset(one_hot = True)
    samples, randperm = shuffle(logit_transform.transform(samples))
    labels,_ = shuffle(labels, randperm)
    labels = labels.float()
    pca_transform = PCA(samples, n_components=100)
    samples = pca_transform.transform(samples)
    num_samples = torch.sum(labels, dim = 0)

    r = range(0, 10)
    train_prior_probs = torch.tensor([1 for i in r])*num_samples
    train_prior_probs = train_prior_probs/torch.sum(train_prior_probs)
    test_prior_probs = torch.tensor([1 for i in r])*num_samples
    test_prior_probs = test_prior_probs/torch.sum(test_prior_probs)

    train_samples, train_labels = [],[]
    test_samples, test_labels = [],[]
    for label in range(labels.shape[-1]):
        current_samples = samples[labels[:,label] == 1]
        current_labels = labels[labels[:,label] == 1]
        for_train = current_samples.shape[0]*train_prior_probs[label]/(train_prior_probs[label] + test_prior_probs[label])
        train_samples.append(current_samples[:int(for_train)])
        test_samples.append(current_samples[int(for_train):])
        train_labels.append(current_labels[:int(for_train)])
        test_labels.append(current_labels[int(for_train):])
    train_samples, train_labels = torch.cat(train_samples),torch.cat(train_labels)
    test_samples, test_labels = torch.cat(test_samples),torch.cat(test_labels)
    datasets = (train_samples, train_prior_probs, train_labels, [test_samples], [test_prior_probs], [test_labels], logit_transform, pca_transform)
    torch.save(datasets,"gen_FMNIST_scenario1/datasets_" + str(run) + ".pt")

    sample_dim = train_samples.shape[-1]
    C = train_labels.shape[-1]
    structure = [[ConditionalRealNVPLayer, {'hidden_dims': [80, 80, 80]}] for i in range(6)] + [
        [ConditionalDIFLayer, {'hidden_dims': [32, 32], 'K': 3}] for i in range(1)]
    model_gen = GenerativeClassifier(sample_dim, C, structure)
    print(model_gen.compute_number_params())
    model_gen.gibbs(5, 400, int(70000 / 20), train_samples, train_labels, [test_samples],
                    [test_prior_probs], [test_labels])

    torch.save(model_gen,"gen_FMNIST_scenario1/model_gen_" +str(run) + ".pt")
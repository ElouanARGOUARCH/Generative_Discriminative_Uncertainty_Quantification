from models import *

number_runs = 10

print("Retrieving discriminative results")

list_train_accuracy = []
list_test_accuracy = []
for run in range(number_runs):
    model_disc = torch.load("disc_MNIST_scenario2/model_disc_" + str(run) +".pt")
    train_samples, train_prior_probs, train_labels,list_test_samples, list_test_prior_probs, list_test_labels,logit_transform, pca_transform = torch.load("disc_MNIST_scenario2/datasets_" + str(run) + ".pt")
    print('run ' + str(run))
    train_accuracy = compute_accuracy(model_disc.log_prob(train_samples), train_labels)
    list_train_accuracy.append(train_accuracy)
    print('train accurarcy ' + str(train_accuracy.item()))
    test_accuracy = compute_accuracy(model_disc.log_prob(list_test_samples[0]), list_test_labels[0])
    list_test_accuracy.append(test_accuracy)
    print('test accurarcy ' + str(test_accuracy.item()))
mean_train_accuracy = torch.tensor(list_train_accuracy).mean()
mean_test_accuracy = torch.tensor(list_test_accuracy).mean()
std_train_accuracy = torch.tensor(list_train_accuracy).std()
std_test_accuracy = torch.tensor(list_test_accuracy).std()
print("train accuracy = " + str(mean_train_accuracy) + "+/-" + str(std_train_accuracy))
print("test accuracy = " + str(mean_test_accuracy) + "+/-" + str(std_test_accuracy))


print("Retrieving generative results")

list_train_accuracy = []
list_test_accuracy = []
for run in range(number_runs):
    model_gen = torch.load("gen_MNIST_scenario2/model_gen_" + str(run) +".pt")
    train_samples, train_prior_probs, train_labels,test_samples, test_prior_probs, test_labels,logit_transform, pca_transform = torch.load("gen_MNIST_scenario2/datasets_" + str(run) + ".pt")
    print('run ' + str(run))
    train_accuracy = compute_accuracy(model_gen.log_posterior_prob(train_samples, train_prior_probs), train_labels)
    list_train_accuracy.append(train_accuracy)
    print('train accurarcy ' + str(train_accuracy.item()))
    test_accuracy = compute_accuracy(model_gen.log_posterior_prob(list_test_samples[0], list_test_prior_probs[0]), list_test_labels[0])
    list_test_accuracy.append(test_accuracy)
    print('test accurarcy ' + str(test_accuracy.item()))
mean_train_accuracy = torch.tensor(list_train_accuracy).mean()
mean_test_accuracy = torch.tensor(list_test_accuracy).mean()
std_train_accuracy = torch.tensor(list_train_accuracy).std()
std_test_accuracy = torch.tensor(list_test_accuracy).std()
print("train accuracy = " + str(mean_train_accuracy) + "+/-" + str(std_train_accuracy))
print("test accuracy = " + str(mean_test_accuracy) + "+/-" + str(std_test_accuracy))

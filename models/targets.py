import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


def shuffle(tensor, randperm=None):
    if randperm is None:
        randperm = torch.randperm(tensor.shape[0])
    return tensor[randperm], randperm

def get_MNIST_dataset(one_hot = False,repository = 'C:\\Users\\Elouan\\PycharmProjects\\models\\targets\\data', visual = False):
    mnist_trainset = datasets.MNIST(root=repository, train=True,
                                    download=True, transform=None)
    mnist_testset = datasets.MNIST(root=repository, train=False,
                                   download=True, transform=None)
    train_labels = mnist_trainset.targets
    test_labels = mnist_testset.targets
    temp_train = mnist_trainset.data.flatten(start_dim=1).float()
    train_samples = (temp_train + torch.rand_like(temp_train))/256
    temp_test = mnist_testset.data.flatten(start_dim=1).float()
    test_samples = (temp_test + torch.rand_like(temp_test))/256
    if visual:
        for i in range(9):
            # define subplot
            plt.subplot(330 + 1 + i)
            # plot raw pixel data
            plt.imshow(train_samples[train_labels==i,:][0].reshape(28,28))
        # show the figure
        plt.show()
    if one_hot:
        return torch.cat([train_samples, test_samples], dim = 0), torch.nn.functional.one_hot(torch.cat([train_labels,test_labels], dim = 0))
    else:
        return torch.cat([train_samples, test_samples], dim = 0), torch.cat([train_labels,test_labels], dim = 0)

_ = get_MNIST_dataset()

def get_FashionMNIST_dataset(one_hot = False,repository = 'C:\\Users\\Elouan\\PycharmProjects\\models\\targets\\data', visual = False):
    fmnist_trainset = datasets.FashionMNIST(root=repository, train=True,
                                            download=True, transform=None)
    fmnist_testset = datasets.FashionMNIST(root=repository, train=False,
                                           download=True, transform=None)
    train_labels = fmnist_trainset.targets
    test_labels = fmnist_testset.targets
    temp_train = fmnist_trainset.data.flatten(start_dim=1).float()
    train_samples = (temp_train + torch.rand_like(temp_train))/256
    temp_test = fmnist_testset.data.flatten(start_dim=1).float()
    test_samples = (temp_test + torch.rand_like(temp_test))/256
    if visual:
        for i in range(9):
            # define subplot
            plt.subplot(330 + 1 + i)
            # plot raw pixel data
            plt.imshow(train_samples[train_labels==i,:][0].reshape(28,28))
        # show the figure
        plt.show()
    if one_hot:
        return torch.cat([train_samples, test_samples], dim = 0), torch.nn.functional.one_hot(torch.cat([train_labels,test_labels], dim = 0))
    else:
        return torch.cat([train_samples, test_samples], dim = 0), torch.cat([train_labels,test_labels], dim = 0)


import os
import torch
import torchvision.transforms as transforms
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.interactive(False)

# Constructing standard transformations for datasets

transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    cifar10_normalization()
    ])

transform_cifar100=transforms.Compose([
    transforms.ToTensor(),
    cifar10_normalization()
    ])

transform_mnist=transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

transform_fmnist=transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,)),
    ])

TRANSFORMS_DICT = {
    'cifar10'   : transform_cifar10,
    'cifar100'  : transform_cifar100,
    'mnist'     : transform_mnist,
    'fmnist'    : transform_fmnist,
    }

def visualize_latent(model, active_dataset, cfg, run_no, verbose=0):
    """
    This method visualize the mean and [mean, log variance] of latent space on 2D scatter plot 
    The method for embedding into 2 dimensions is tSNE
    The results are saved into save/results/<experiment_name>/latent_visual_mu_<logvar>_<run_no>
    """

    if not os.path.exists(f'save/results/{cfg.experiment_name}'):
        os.makedirs(f'save/results/{cfg.experiment_name}')
    
    nr_of_samples = 2500
    device = cfg.device

    all_DL = active_dataset.get_loader('train', batch_size=nr_of_samples, shuffle=True)
    all_iter = iter(all_DL)
    x, y = next(all_iter)
    x = x.to(device)

    latent = model.latent_param(x).cpu() # taking both mean and log variance from latent space

    X_embedded_mu_logvar = TSNE(n_components=2, early_exaggeration=70, perplexity=30,
                      learning_rate=500, init='pca', n_iter=5000, n_iter_without_progress=300, verbose=verbose,
                      random_state=0).fit_transform(torch.reshape(latent, (nr_of_samples, -1)))

    fig, ax = plt.subplots()
    plt.title(f'Run no: {run_no} Latent Space of Mu and Logvar')
    cmap = plt.cm.get_cmap('tab10', 10)
    plt.scatter(x=X_embedded_mu_logvar[:, 0], y=X_embedded_mu_logvar[:, 1], c=y, s=20, cmap=cmap)
    plt.colorbar()
    plt.savefig(f'save/results/{cfg.experiment_name}/latent_visual_mu_logvar_{run_no}.png')

    mu = model.latent_mu(x).cpu() # taking mean from latent space

    X_embedded_mu = TSNE(n_components=2, early_exaggeration=70, perplexity=30,
                      learning_rate=500, init='pca', n_iter=5000, n_iter_without_progress=300, verbose=verbose,
                      random_state=0).fit_transform(mu)

    fig, ax = plt.subplots()
    plt.title(f'Run no: {run_no} Latent Space of Mu')
    cmap = plt.cm.get_cmap('tab10', 10)
    plt.scatter(x=X_embedded_mu[:, 0], y=X_embedded_mu[:, 1], c=y, s=20, cmap=cmap)
    plt.colorbar()
    plt.savefig(f'save/results/{cfg.experiment_name}/latent_visual_mu_{run_no}.png')

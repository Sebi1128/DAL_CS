from tqdm import tqdm
import wandb
import torch
from copy import deepcopy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.interactive(True)



def visualize_latent(model, active_dataset, cfg):
    nr_of_samples = 2500
    device = cfg.device

    all_DL = active_dataset.get_loader('train', batch_size=nr_of_samples, shuffle=True)
    all_iter = iter(all_DL)
    x, y = next(all_iter)
    x = x.to(device)

    latent = model.latent_param(x).cpu()

    X_embedded_mu_logvar = TSNE(n_components=2, early_exaggeration=70, perplexity=30,
                      learning_rate=500, init='pca', n_iter=5000, n_iter_without_progress=300, verbose=5,
                      random_state=0).fit_transform(torch.reshape(latent, (nr_of_samples, -1)))

    fig, ax = plt.subplots()
    plt.title('Latent space of mu and logvar')
    cmap = plt.cm.get_cmap('tab10', 10)
    plt.scatter(x=X_embedded_mu_logvar[:, 0], y=X_embedded_mu_logvar[:, 1], c=y, s=20, cmap=cmap)
    plt.colorbar()
    plt.savefig('save/results/latent_visual_mu_logvar.png')

    mu = model.latent_mu(x).cpu()

    X_embedded_mu = TSNE(n_components=2, early_exaggeration=70, perplexity=30,
                      learning_rate=500, init='pca', n_iter=5000, n_iter_without_progress=300, verbose=5,
                      random_state=0).fit_transform(mu)

    fig, ax = plt.subplots()
    plt.title('Latent space of mu')
    cmap = plt.cm.get_cmap('tab10', 10)
    plt.scatter(x=X_embedded_mu[:, 0], y=X_embedded_mu[:, 1], c=y, s=20, cmap=cmap)
    plt.colorbar()
    plt.savefig('save/latent_visual_mu.png')



def epoch_run(model, sampler, active_dataset, run_no, model_writer, cfg):
    pbar = tqdm(range(cfg.n_epochs))
    pbar.set_description("training")

    acc_best_valid = -1
    loss_best_valid = 1e10
    for epoch_no in pbar:
        train_loss = train_epoch(
            model,
            sampler,
            active_dataset,
            batch_size=cfg.batch_size,
            device=cfg.device,
            train_vae=cfg.embedding['train_vae']
        )
        valid_loss, valid_acc = validate_epoch(
            model,
            sampler,
            active_dataset,
            batch_size=cfg.batch_size,
            device=cfg.device,
            train_vae=cfg.embedding['train_vae']
        )

        info_text = f"{run_no+1}|C|R|SE|SS Train:|" + '|'.join([f"{val:.5f}" for val in train_loss.values()]) + '|'
        info_text += f"  Valid:|" + '|'.join([f"{val:.5f}" for val in valid_loss.values()]) + '|'
        info_text += f" Task Acc: {valid_acc:3.3f}%"
        pbar.set_description(info_text, refresh=True)

        train_loss['epoch'] = epoch_no
        train_loss['run_no'] = run_no
        valid_loss['epoch'] = epoch_no
        valid_loss['run_no'] = run_no

        wandb_data = dict()
        wandb_data.update(train_loss)
        wandb_data.update(valid_loss)
        wandb_data.update({"valid_accuracy": valid_acc, "epoch": epoch_no, "run_no": run_no})

        wandb.log(wandb_data)

        # No need for evaluating this, we can observe it on wandb
        if valid_acc > acc_best_valid:
            model_writer.write(model, 'best_acc_')
            #if sampler.trainable:
            #    model_writer.write(sampler, 'sampler_' + name)
            acc_best_valid = valid_acc

        if valid_loss['classification_loss_val'] < loss_best_valid:
            model_writer.write(model, 'best_loss_')
            loss_best_valid = valid_loss['classification_loss_val']

    test_acc_last_epoch = test_epoch(model, active_dataset, batch_size=cfg.batch_size, device=cfg.device, model_writer=model_writer)
    test_acc_best_acc = test_epoch(model, active_dataset, batch_size=cfg.batch_size, device=cfg.device, model_writer=model_writer, load_prefix='best_acc_')
    test_acc_best_loss = test_epoch(model, active_dataset, batch_size=cfg.batch_size, device=cfg.device, model_writer=model_writer, load_prefix='best_loss_')
    wandb.log({ "test_acc_last_epoch"   : test_acc_last_epoch, 
                "test_acc_best_acc"     : test_acc_best_acc,
                "test_acc_best_loss"    : test_acc_best_loss,
                "run_no": run_no})

    #print(f"Best Classification Loss \t{c_best_valid_loss} with Epoch No {c_best_epoch_no} for Run {run_no}")
    #print(f"Final Reconstruction Loss \t{r_best_valid_loss} with Epoch No {r_best_epoch_no} for Run {run_no}")


def train_epoch(model, sampler, active_data, batch_size, device, train_vae=True):

    model.train()
    torch.set_grad_enabled(True)
    
    iter_schedule = active_data.get_itersch(uniform=False)

    lbld_DL = active_data.get_loader('labeled', batch_size=batch_size)
    unlbld_DL = active_data.get_loader('unlabeled', batch_size=batch_size)
    all_DL = active_data.get_loader('train', batch_size=batch_size)

    lbl_iter = iter(lbld_DL)
    unlbl_iter = iter(unlbld_DL)
    all_iter = iter(all_DL)

    n_epochs = len(active_data.trainset) // batch_size
    c_losses = list()
    r_losses = list()
    se_losses = list()
    ss_losses = list()

    pbar = tqdm(iter_schedule[:n_epochs], leave=False)
    pbar.set_description("model epoch")
    for is_labeled in pbar:
        if is_labeled:
            x, t = next(lbl_iter)
            x = x.to(device)
            t = t.to(device)
            c = model.classify(x)
            loss = model.c_loss(c, t)
            c_losses.append(float(loss))

            model.optimizer_classifier.zero_grad()
            loss.backward()
            model.optimizer_classifier.step()

            if train_vae:
                if sampler.trainable:
                    x_unlabeled, _ = next(unlbl_iter)
                    x_unlabeled = x_unlabeled.to(device)
                    mu_labeled = model.latent_param(x_unlabeled)[..., 0]
                    mu_unlabeled = model.latent_param(x_unlabeled)[..., 0]
                    sampler_in = (mu_labeled, mu_unlabeled)
                    sampler_out = sampler(sampler_in)
                    loss = sampler.model_loss(sampler_out)
                    se_losses.append(float(loss))

                    model.optimizer_embedding.zero_grad()
                    loss.backward()
                    model.optimizer_embedding.step()
        else:
            if train_vae:
                x, _ = next(all_iter)
                x = x.to(device)
                r, latent = model.reconstruct(x)
                loss = model.r_loss(r.flatten(), x.flatten(), *latent[1:])['loss']
                r_losses.append(float(loss))

                model.optimizer_embedding.zero_grad()
                loss.backward()
                model.optimizer_embedding.step()

    # sampler (discriminator) is trained separately (unlike vaal) from generator (as suggested for GAN)
    if sampler.trainable:
        lbld_DL = active_data.get_loader('labeled', batch_size=batch_size)
        unlbld_DL = active_data.get_loader('unlabeled', batch_size=batch_size)

        sampler.train()
        pbar_sub_ep = tqdm(range(sampler.n_sub_epochs), leave=False)
        for _ in pbar_sub_ep:
            lbl_iter = iter(lbld_DL)
            unlbl_iter = iter(unlbld_DL)
            pbar_step = tqdm(range(min(len(lbl_iter), len(unlbl_iter))), leave=False)
            pbar.set_description("sampler epoch")
            for _ in pbar_step:
                x_labeled, _ = next(lbl_iter)
                x_unlabeled, _ = next(unlbl_iter)
                x_labeled = x_labeled.to(device)
                x_unlabeled = x_unlabeled.to(device)

                mu_labeled = model.latent_param(x_labeled)[..., 0]
                mu_unlabeled = model.latent_param(x_unlabeled)[..., 0]

                sampler_in = (mu_labeled, mu_unlabeled)
                sampler_out = sampler(sampler_in)
                loss = sampler.sampler_loss(sampler_out)
                ss_losses.append(float(loss))

                sampler.optimizer.zero_grad()
                loss.backward()
                sampler.optimizer.step()

    result = {
        'classification_loss_train': torch.mean(torch.tensor(c_losses)),
        'reconstruction_loss_train': torch.mean(torch.tensor(r_losses)),
        'sampling_embedding_loss_train': torch.mean(torch.tensor(se_losses)),
        'sampling_sampler_loss_train': torch.mean(torch.tensor(ss_losses))
    }

    return result

        
def validate_epoch(model, sampler, active_data, batch_size, device, train_vae=True):

    model.eval()
    torch.set_grad_enabled(False)

    valid_DL = active_data.get_loader('validation', batch_size=batch_size)

    c_losses = list()
    r_losses = list()

    correct = 0
    total = 0
    for x, t in valid_DL:
        x = x.to(device)
        t = t.to(device)
        c = model.classify(x)
        loss = model.c_loss(c, t)
        c_losses.append(loss)

        correct += (c.argmax(1) == t).sum()
        total += len(t)

        if train_vae:
            r, latent = model.reconstruct(x)
            loss = model.r_loss(r.flatten(), x.flatten(), *latent[1:])['loss']
            r_losses.append(loss)

    result = {
        'classification_loss_val': torch.mean(torch.tensor(c_losses)),
        'reconstruction_loss_val': torch.mean(torch.tensor(r_losses)),
    }
    return result, torch.true_divide(correct, total) * 100
    #return result, correct / total * 100


def test_epoch(model_actual, active_data, batch_size, device, model_writer, load_prefix=None):
    test_DL = active_data.get_loader('test', batch_size=batch_size)

    model = deepcopy(model_actual)

    if load_prefix is not None:
        model_writer.load(model, prefix=load_prefix)
    
    correct = 0
    total = 0

    for x, t in test_DL:
        x = x.to(device)
        t = t.to(device)
        c = model.classify(x)

        correct += (c.argmax(1) == t).sum()
        total += len(t)

    return correct / total * 100


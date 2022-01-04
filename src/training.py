from tqdm import tqdm
import wandb
import torch
from copy import deepcopy
from src.training_utils import visualize_latent


def run(model, sampler, active_dataset, run_no, model_writer, cfg):
    pbar = tqdm(range(cfg.n_epochs))
    pbar.set_description("training")

    acc_best_valid = -1 # highest accuracy initializer
    loss_best_valid = 1e10 # lowest accuracy initializer

    for epoch_no in pbar: 
        train_loss_dict = train_epoch( # training of epoch
            model, # model to train
            sampler, # sampler to train
            active_dataset, # dataset to be used
            batch_size=cfg.batch_size,
            device=cfg.device,
            train_vae=cfg.embedding['train_vae']
        )
        valid_loss_dict, valid_acc = validate_epoch( # evaluating the model success
            model, # model to assess accuracy/loss
            active_dataset, # dataset to be used
            batch_size=cfg.batch_size,
            device=cfg.device,
            train_vae=cfg.embedding['train_vae']
        )

        # Info text using tqdm
        info_text = f"{run_no+1}|C|R|SE|SS Train:|" + '|'.join([f"{val:.5f}" for val in train_loss_dict.values()]) + '|'
        info_text += f"  Valid:|" + '|'.join([f"{val:.5f}" for val in valid_loss_dict.values()]) + '|'
        info_text += f" Task Acc: {valid_acc:3.3f}%"
        pbar.set_description(info_text, refresh=True)

        # the dictionary containing information of the step
        wandb_data = dict() 
        wandb_data.update(train_loss_dict)
        wandb_data.update(valid_loss_dict)
        wandb_data.update({"valid_accuracy": valid_acc, "epoch": epoch_no, "run_no": run_no})

        wandb.log(wandb_data)

        # save the model if it has higher validation accuracy of all epochs with prefix best_acc_ 
        if valid_acc > acc_best_valid:
            model_writer.write(model, 'best_acc_') # saving model parameters to
            # save/param/<date>_<experiment_name>_<seed>_<W&B_ID>/best_acc_weights.pth
            acc_best_valid = valid_acc # defining new highest accuracy

        # save the model if it has lower validation loss of all epochs with prefix best_loss_
        if valid_loss_dict['classification_loss_val'] < loss_best_valid:
            model_writer.write(model, 'best_loss_') # saving model parameters to 
            # save/param/<date>_<experiment_name>_<seed>_<W&B_ID>/best_loss_weights.pth
            loss_best_valid = valid_loss_dict['classification_loss_val'] # defining new lowest loss

    # in the end of the run, three types of test accuracy is calculated with the parameters of
    # 1) last epoch, best validation accuracy, best validation loss
    test_acc_last_epoch = test_epoch(model, active_dataset, batch_size=cfg.batch_size, 
                                     device=cfg.device, model_writer=model_writer)
    test_acc_best_acc = test_epoch(model, active_dataset, batch_size=cfg.batch_size, 
                                   device=cfg.device, model_writer=model_writer, load_prefix='best_acc_')
    test_acc_best_loss = test_epoch(model, active_dataset, batch_size=cfg.batch_size, 
                                    device=cfg.device, model_writer=model_writer, load_prefix='best_loss_')
    wandb.log({ "test_acc_last_epoch"   : test_acc_last_epoch, 
                "test_acc_best_acc"     : test_acc_best_acc,
                "test_acc_best_loss"    : test_acc_best_loss,
                "run_no": run_no})

    if cfg.visual_latent == True: # if true, saves the tSNE embeddıngs of latent spaces to
        # save/results/<experiment_name>/latent_visual_mu_<logvar>_<run_no>
        visualize_latent(model, active_dataset, cfg, run_no)

def train_epoch(model, sampler, active_data, batch_size, device, train_vae=True):

    # enabling the training mode
    model.train()
    torch.set_grad_enabled(True)

    # getting iteration schedule 
    iter_schedule = active_data.get_itersch(uniform=False)

    lbld_DL = active_data.get_loader('labeled', batch_size=batch_size)
    unlbld_DL = active_data.get_loader('unlabeled', batch_size=batch_size)
    all_DL = active_data.get_loader('train', batch_size=batch_size)

    lbl_iter = iter(lbld_DL)
    unlbl_iter = iter(unlbld_DL)
    all_iter = iter(all_DL)

    n_iters = len(active_data.trainset) // batch_size
    c_losses = list()
    r_losses = list()
    se_losses = list()
    ss_losses = list()

    pbar = tqdm(iter_schedule[:n_iters], leave=False)
    pbar.set_description("iterations of epoch")
    for is_labeled in pbar:
        if is_labeled:
            try:
                x, t = next(lbl_iter)
            except:
                continue
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

        
def validate_epoch(model, active_data, batch_size, device, train_vae=True):

    # enabling evaluation mode
    model.eval()
    torch.set_grad_enabled(False)

    # getting validation dataset data loader
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


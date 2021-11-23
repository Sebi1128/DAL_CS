from tqdm import tqdm
import wandb
import torch

def epoch_run(model, sampler, active_dataset, optimizer, run_no, model_writer, cfg):
    pbar = tqdm(range(cfg.n_epochs))
    #pbar = tqdm(range(1)) # debug
    pbar.set_description("validation")

    acc_best_valid = -1

    for epoch_no in pbar:
        train_loss = train_epoch(model, sampler, active_dataset, optimizer, batch_size=cfg.batch_size,
                                 device=cfg.device)
        valid_loss, valid_acc = validate_epoch(model, sampler, active_dataset, batch_size=cfg.batch_size,
                                                   device=cfg.device)

        info_text = f"{run_no+1}|C|R|SE|SS Train:|" + '|'.join([f"{val:.5f}" for val in train_loss.values()]) + '|'
        info_text += f"  Valid:|" + '|'.join([f"{val:.5f}" for val in valid_loss.values()]) + '|'
        info_text += f" Task Acc: {valid_acc:3.3f}%"
        pbar.set_description(info_text, refresh=True)

        train_loss['epoch'] = epoch_no
        train_loss['run_no'] = run_no
        valid_loss['epoch'] = epoch_no
        valid_loss['run_no'] = run_no
        wandb.log(train_loss)
        wandb.log(valid_loss)
        wandb.log({"valid accuracy": valid_acc, "epoch": epoch_no, "run_no": run_no})

        # No need for evaluating this, we can observe it on wandb
        #if r_valid_loss < r_best_valid_loss:
        #    r_best_epoch_no = epoch_no
        #if c_valid_loss < c_best_valid_loss:
        #    c_best_epoch_no = epoch_no
        if valid_acc > acc_best_valid:
            name = f'run_{run_no:01d}_epo_{epoch_no:04d}_'
            model_writer.write(model, 'model_'+name)
            if sampler.trainable:
                model_writer.write(sampler, 'sampler_' + name)

    test_acc = test_epoch(model, active_dataset, batch_size=cfg.batch_size, device=cfg.device)
    wandb.log({"test accuracy": test_acc, "run_no": run_no})

    #print(f"Best Classification Loss \t{c_best_valid_loss} with Epoch No {c_best_epoch_no} for Run {run_no}")
   # print(f"Final Reconstruction Loss \t{r_best_valid_loss} with Epoch No {r_best_epoch_no} for Run {run_no}")


def train_epoch(model, sampler, active_data, optimizer, batch_size, device):

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
    #n_epochs = 2 # debug
    c_losses = list()
    r_losses = list()
    se_losses = list()
    ss_losses = list()

    pbar = tqdm(iter_schedule[:n_epochs], leave=False)
    pbar.set_description("training")
    for is_labeled in pbar:
        if is_labeled:
            x, t = next(lbl_iter)
            x = x.to(device)
            t = t.to(device)
            c = model.classify(x)
            loss = model.c_loss(c, t)
            c_losses.append(loss)

            if sampler.trainable:
                x_unlabeled, _ = next(unlbl_iter)
                x_unlabeled.to(device)
                r_labeled = model.latent(x)
                r_unlabeled = model.latent(x_unlabeled)
                sampler_in = (r_labeled, r_unlabeled)
                sampler_out = sampler(sampler_in)
                se_loss = sampler.model_loss(sampler_out)
                se_losses.append(se_loss)
                loss += se_loss
        else:
            x, _ = next(all_iter)
            x = x.to(device)
            r, latent = model.reconstruct(x)
            loss = model.r_loss(r.flatten(), x.flatten(), *latent[1:])['loss']
            r_losses.append(loss)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if sampler.trainable:
        lbld_DL = active_data.get_loader('labeled', batch_size=batch_size)
        unlbld_DL = active_data.get_loader('unlabeled', batch_size=batch_size)
        lbl_iter = iter(lbld_DL)
        unlbl_iter = iter(unlbld_DL)

        sampler.train()
        for sub_epoch in range(sampler.n_sub_epochs):
            for step in range(min(len(lbl_iter), len(unlbl_iter))):
                x, _ = next(lbl_iter)
                x_unlabeled, _ = next(unlbl_iter)
                x = x.to(device)
                x_unlabeled = x_unlabeled.to(device)

                r_labeled = model.latent(x)
                r_unlabeled = model.latent(x_unlabeled)

                sampler_in = (r_labeled, r_unlabeled)
                sampler_out = sampler(sampler_in)
                loss = sampler.sampler_loss(sampler_out)
                ss_losses.append(loss)

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

        
def validate_epoch(model, sampler, active_data, batch_size, device):

    model.eval()
    torch.set_grad_enabled(False)

    valid_DL = active_data.get_loader('validation', batch_size=batch_size)

    c_losses = list()
    r_losses = list()
    se_losses = list()
    ss_losses = list()

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

        r, latent = model.reconstruct(x)
        loss = model.r_loss(r.flatten(), x.flatten(), *latent[1:])['loss']
        r_losses.append(loss)

        if sampler.trainable:
            sampler_in = (r_labeled, r_unlabeled)
            sampler_out = sampler(sampler_in)
            loss = sampler.model_loss(sampler_out)
            se_losses.append(loss)

            loss = sampler.sampler_loss(sampler_out)
            ss_losses.append(loss)

        #break # DEBUG

    result = {
        'classification_loss_val': torch.mean(torch.tensor(c_losses)),
        'reconstruction_loss_val': torch.mean(torch.tensor(r_losses)),
        'sampling_embedding_loss_val': torch.mean(torch.tensor(se_losses)),
        'sampling_sampler_loss_val': torch.mean(torch.tensor(ss_losses))
    }

    return result, correct / total * 100


def test_epoch(model, active_data, batch_size, device):
    test_DL = active_data.get_loader('test', batch_size=batch_size)

    correct = 0
    total = 0

    for x, t in test_DL:
        x = x.to(device)
        t = t.to(device)
        c = model.classify(x)

        correct += (c.argmax(1) == t).sum()
        total += len(t)

        #break # debug

    return correct / total * 100


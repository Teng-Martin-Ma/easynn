from easynn.model import Model, init_params
from easynn.data import EasyData
from easynn.logging import Logger
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import yaml
import os
import numpy as np
from tqdm import tqdm
import time


def train(params_file):
    """Train the NN model.

    Args:
        params_file (str): path to the params file
    """

    with open(params_file, 'r') as f:
        p = yaml.full_load(f)

    if p['Device'] == 'cpu':
        device = torch.device('cpu')
    elif p['Device'] == 'gpu':
        device = torch.device('cuda:0')
    else:
        raise ValueError('Invalid device')

    if p['Dtype'] == 'float64':
        dtype = torch.float64
    elif p['Dtype'] == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Invalid dtype')

    torch.manual_seed(p['Seed'])
    torch.cuda.manual_seed(p['Seed'])
    np.random.seed(p['Seed'])

    # Logging
    os.makedirs(p['WorkDir'], exist_ok=True)
    log = Logger(p['WorkDir']+'LOG')

    # Read data
    ED = EasyData(params_file)
    if p['Transform']:
        trans_start = time.time()
        log.info('Transforming data...')
        ED.transform_data()
        trans_time = time.time() - trans_start
        log.info(f"Data transformed. The total time is {trans_time:.4f} seconds.")

    data_train = ED.load_data(train=True)
    ntrain = torch.cat(data_train['ID']).shape[-1]
    dl_train = DataLoader(torch.arange(ntrain),
                          batch_size=p['BatchSize'],
                          shuffle=True)
    data_valid = ED.load_data(train=False)
    nvalid = torch.cat(data_valid['ID']).shape[-1]
    dl_valid = DataLoader(torch.arange(nvalid),
                          batch_size=10,
                          shuffle=False)
    if p['Scale']['Scale']:
        data_scale = {
            'E': data_train['E'] + data_valid['E'],
            'F': data_train['F'] + data_valid['F'],
            'G': data_train['G'] + data_valid['G']}
        scale = ED.scale_data(data_scale)
        torch.save(scale, p['WorkDir']+'scale.pt')

    # Initialize or load the model
    model = Model(data_train['G'][0].shape[-1], p['HiddenLayers'])
    if dtype == torch.float64:
        model.double()
    if p['Resume']:
        model.load_state_dict(torch.load(p['WorkDir']+'best_model.pt'))
    else:
        model.apply(init_params)
    model.to(device=device)
    log.info(f"Model has {model.count_params()} trainable parameters.")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=p['LearningRate'])
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=p['LearningRateDecay']['factor'],
        patience=p['LearningRateDecay']['patience'])

    # Train the model
    os.environ["WANDB_NOTEBOOK_NAME"] = "EasyNN"
    wandb.init(project=p['Project'], name=p['TaskName'], resume=p['Resume'],
               anonymous="allow")
    Ce, Cf, Cw = p['LossFn']['Ce'], p['LossFn']['Cf'], p['LossFn']['Cw']
    ESCOUNT = 0  # Early stopping counter
    train_start = time.time()
    log.info(f"Training data... started at {time.ctime(train_start)}")
    log.info('Epoch | Train Loss | Valid Loss |')
    for ep in tqdm(range(p['MaxEpochs'])):
        if optimizer.param_groups[0]['lr'] < float(p['LearningRateDecay']['min_lr']):
            log.info("Learning rate is below the minimum value")
            log.info("Training stopped")
            log.info(f"Total training time: {time.time()-train_start:.4f} s")
            break

        # Train
        model.train()
        loss_train = 0
        for batch in dl_train:
            optimizer.zero_grad()
            loss_e, loss_f, N = 0, 0, 0
            for i in batch:
                G = data_train['G'][i].to(device=device, dtype=dtype)
                dGdR = data_train['dGdR'][i].to(device=device, dtype=dtype)
                E = data_train['E'][i].to(device=device, dtype=dtype)
                F = data_train['F'][i].to(device=device, dtype=dtype)
                N += data_train['N'][i]

                G.requires_grad_(True)
                if p['Scale']['Scale']:
                    G = (G - scale['G_mean']) / scale['G_std']
                    E = (E - scale['E_mean']) / scale['E_std']
                    F = (F - scale['F_mean']) / scale['F_std']

                E_pred = model(G).sum()
                dEdG = torch.autograd.grad(
                    E_pred, G, create_graph=True, retain_graph=True)[0]
                F_pred = - torch.einsum('ij, ijkw -> kw', dEdG, dGdR)

                loss_e += (E_pred - E)**2
                loss_f += ((F_pred - F)**2).sum()
                G.grad = None

            loss = Ce * loss_e/len(batch) + Cf * loss_f/(3*N)
            loss.backward()
            loss_train += loss.item()
            optimizer.step()

        loss_train /= len(dl_train)

        # Validate
        model.eval()
        loss_valid = 0
        for batch in dl_valid:
            loss_e, loss_f, N = 0, 0, 0
            for i in batch:
                G = data_valid['G'][i].to(device=device, dtype=dtype)
                dGdR = data_valid['dGdR'][i].to(device=device, dtype=dtype)
                E = data_valid['E'][i].to(device=device, dtype=dtype)
                F = data_valid['F'][i].to(device=device, dtype=dtype)
                N += data_valid['N'][i]

                G.requires_grad_(True)
                if p['Scale']['Scale']:
                    G = (G - scale['G_mean']) / scale['G_std']
                    E = (E - scale['E_mean']) / scale['E_std']
                    F = (F - scale['F_mean']) / scale['F_std']

                E_pred = model(G).sum()
                dEdG = torch.autograd.grad(E_pred, G, create_graph=True)[0]
                F_pred = - torch.einsum('ij, ijkw -> kw', dEdG, dGdR)

                loss_e += (E_pred - E)**2
                loss_f += ((F_pred - F)**2).sum()
                G.grad = None

            loss = Ce * loss_e/len(batch) + Cf * loss_f/(3*N)
            loss_valid += loss.item()

        loss_valid /= len(dl_valid)

        # log to wandb
        wandb.log({'ep': ep,
                   'loss_train': loss_train,
                   'loss_valid': loss_valid,
                   'cumulative_time': time.time() - train_start,
                   'lr': optimizer.param_groups[0]['lr']})

        # Save the best model
        if ep == 0:
            best_loss = loss_valid
        if loss_valid < best_loss:
            best_loss = loss_valid
            torch.save(model, p['WorkDir']+'best_model.pt')
        if ep % 100 == 0:
            torch.save(model, p['WorkDir']+'checkpoint.pt')
            log.info(f"{ep:06d} {loss_train:8.4f} {loss_valid:8.4f}")
        if ep > p['MaxEpochs']:
            log.info(f"Not converged after {p['MaxEpochs']} epochs!")
            log.info("Please increase the number of epochs!")
            log.info(f"Train loss: {loss_train:.4f} Valid loss: {loss_valid:.4f}")
            log.info(f"Total time: {time.time() - train_start:.4f} s.")
            log.close()
            wandb.finish()
            break

        # Early stopping
        if loss_valid - best_loss > p['EarlyStopping']['threshold']:
            ESCOUNT += 1
        else:
            ESCOUNT = 0
        if ESCOUNT > p['EarlyStopping']['patience']:
            log.info("The validation loss has not improved for 25 epochs.")
            log.info(f"Training stopped at epoch {ep}.")
            log.info(f"Train loss: {loss_train:.4f} Valid loss: {loss_valid:.4f}")
            log.info(f"Total time: {time.time() - train_start:.4f} s.")
            log.close()
            wandb.finish()
            break
        scheduler.step(loss_valid)

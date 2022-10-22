from easynn.model import Model, init_params
from easynn.data import EASYDATA
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import yaml
import os
import numpy as np
from tqdm import tqdm
import time
import logging


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

    # Read data
    ED = EASYDATA(params_file)
    if p['Transform']:
        ED.transform_data()
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

    # Initialize or load the model
    model = Model(data_train['G'][0].shape[-1], p['HiddenLayers'])
    if dtype == torch.float64:
        model.double()
    if p['ReStart']:
        model.apply(init_params)
    else:
        model.load_state_dict(torch.load(p['WorkDir']+'best_model.pt'))
    model.to(device=device)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=p['LearningRate'])
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=p['LearningRateDecay']['factor'],
        patience=p['LearningRateDecay']['patience'])

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=p['WorkDir']+p['LogFile'])

    # Train the model
    os.makedirs(p['WorkDir'], exist_ok=True)
    os.environ["WANDB_NOTEBOOK_NAME"] = "EasyNN"
    wandb.init(project=p['Project'], name=p['TaskName'], resume=p['ReStart'],
               anonymous="allow")
    wandb.login()
    Ce, Cf, Cw = p['LossFn']['Ce'], p['LossFn']['Cf'], p['LossFn']['Cw']
    ESCOUNT = 0  # Early stopping counter
    start = time.time()
    logging.info(f"Training started at {time.ctime(start)}")
    for ep in tqdm(range(p['MaxEpochs'])):
        if optimizer.param_groups[0]['lr'] < float(p['LearningRateDecay']['min_lr']):
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
                   'cumulative_time': time.time() - start,
                   'lr': optimizer.param_groups[0]['lr']})

        # Save the best model
        if ep == 0:
            best_loss = loss_valid
        if loss_valid < best_loss:
            best_loss = loss_valid
            torch.save(model.state_dict(), p['WorkDir']+'best_model.pt')
            logging.info(f"Best model saved at epoch {ep}!")
        if ep > p['MaxEpochs']:
            logging.info(f"Not converged after {p['MaxEpochs']} epochs!")
            logging.info("Please increase the number of epochs!")
            wandb.finish()
            logging.info(f"Total time: {time.time() - start:.2f} s.")
            break

        # Early stopping
        if loss_valid - best_loss > p['EarlyStopping']['threshold']:
            ESCOUNT += 1
        else:
            ESCOUNT = 0
        if ESCOUNT > p['EarlyStopping']['patience']:
            logging.info("The validation loss has not improved for 25 epochs.")
            logging.info(f"Training stopped at epoch {ep}.")
            wandb.finish()
            logging.info(f"Total time: {time.time() - start:.2f} s.")
            break
        scheduler.step(loss_valid)

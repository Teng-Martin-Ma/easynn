from easynn.model import Model, init_params
from easynn.data import EasyData
from easynn.logging import Logger
from easynn.descriptor import ACSF
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

    device = torch.device(p['Device'])
    if p['Dtype'] == 'float64':
        dtype = torch.float64
    elif p['Dtype'] == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Invalid dtype!')

    torch.manual_seed(p['Seed'])
    torch.cuda.manual_seed(p['Seed'])
    np.random.seed(p['Seed'])

    # Logging
    if p['WorkDir'][-1] != '/':
        p['WorkDir'] += '/'
    os.makedirs(p['WorkDir'], exist_ok=True)
    log = Logger(p['WorkDir']+p['TaskName']+'/log')
    log.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # Read data
    log.info('Reading data...')
    descriptor = ACSF(
        wd=p['WorkDir']+p['TaskName'],
        elements=p['Elements'],
        rc=p['CutoffRadius'],
        device=device,
        dtype=dtype)
    descriptor.save()
    ED = EasyData(p['DataPath'], descriptor, device, dtype)

    if not os.path.exists(p['WorkDir']+p['TaskName']+'/train.pt'):
        trans_start = time.time()
        log.info('Transforming data...')
        train_images, valid_images = ED.split(p['TrainSize'], shuffle=p['Shuffle'])
        ED.transform(train_images, p['WorkDir']+p['TaskName']+'/train.pt')
        ED.transform(valid_images, p['WorkDir']+p['TaskName']+'/valid.pt')
        trans_time = time.time() - trans_start
        log.info("Data have been transformed.")
        log.info(f"The total time is {trans_time:.4f} seconds.")
    else:
        log.info("Data have already been transformed, extract directly.")

    train_data = torch.load(p['WorkDir']+p['TaskName']+'/train.pt')
    ntrain = len(train_data)
    train_dl = DataLoader(torch.arange(ntrain),
                          batch_size=p['TrainBatchSize'],
                          shuffle=True)
    valid_data = torch.load(p['WorkDir']+p['TaskName']+'/valid.pt')
    nvalid = len(valid_data)
    valid_dl = DataLoader(torch.arange(nvalid),
                          batch_size=p['ValidBatchSize'],
                          shuffle=False)

    # Scale data
    if not os.path.exists(p['WorkDir']+p['TaskName']+'/scale.pt'):
        scale_start = time.time()
        log.info('Scaling data...')
        scale = ED.scale([train_data, valid_data], p['ScaleMethod'])
        torch.save(scale, p['WorkDir']+p['TaskName']+'/scale.pt')
        scale_time = time.time() - scale_start
        log.info("Data have been scaled.")
        log.info(f"The total time is {scale_time:.4f} seconds.")
    else:
        log.info("The data have already been scaled.")
        scale = torch.load(p['WorkDir']+p['TaskName']+'/scale.pt')

    # Initialize or load the model
    model = Model(train_data[0]['G'].shape[-1], p['HiddenLayers'])
    if dtype == torch.float64:
        model.double()
    if p['Resume']:
        log.info(f"Loading model from {p['WorkDir']+p['TaskName']}"+'/best_model.pt')
        model.load_state_dict(torch.load(p['WorkDir']+p['TaskName']+'/best_model.pt'))
    else:
        log.info("Initializing model...")
        model.apply(init_params)
    model.to(device=device)
    log.info(f"Model has {model.count_params()} trainable parameters.")

    # Initialize optimizer and scheduler
    Ce, Cf, Cw = p['LossFn']['Ce'], p['LossFn']['Cf'], p['LossFn']['Cw']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=p['LearningRate'],
        weight_decay=Cw)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=p['LearningRateDecay']['factor'],
        patience=p['LearningRateDecay']['patience'])

    # Train the model
    wandb.init(project=p['Project'], name=p['TaskName'], resume=p['Resume'],
               anonymous="allow")

    ESCOUNT = 0  # Early stopping counter
    train_start = time.time()
    log.info("Training data...")
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
        for batch in train_dl:
            optimizer.zero_grad()
            loss_e, loss_f, Ns = 0, 0, 0
            for i in batch:
                image = train_data[i.item()]
                G = image['G'].to(device=device, dtype=dtype)
                dGdR = image['dGdR'].to(device=device, dtype=dtype)
                E = (image['E'] / image['N']).to(device=device, dtype=dtype)
                F = image['F'].to(device=device, dtype=dtype)
                Ns += image['N']

                G.requires_grad_(True)
                if p['ScaleMethod'] == 'Minmax':
                    sG = (G-scale['G_min']) / (scale['G_max']-scale['G_min'])
                    E = (E-scale['E_min']) / (scale['E_max']-scale['E_min'])
                    F = (F-scale['F_min']) / (scale['F_max']-scale['F_min'])
                elif p['ScaleMethod'] == 'Standard':
                    sG = (G - scale['G_mean']) / scale['G_std']
                    E = (E - scale['E_mean']) / scale['E_std']
                    F = (F - scale['F_mean']) / scale['F_std']

                E_pred = model(sG).sum()
                dEdG = torch.autograd.grad(
                    E_pred, G, create_graph=True, retain_graph=True)[0]
                F_pred = - torch.einsum('ij, ijkw -> kw', dEdG, dGdR)

                loss_e += (E_pred - E)**2
                loss_f += ((F_pred - F)**2).sum()
                G.grad = None

            loss = Ce * loss_e/len(batch) + Cf * loss_f/(3*Ns)
            loss.backward()
            loss_train += loss.item()
            optimizer.step()

        loss_train /= len(train_dl)

        # Validate
        model.eval()
        loss_valid = 0
        for batch in valid_dl:
            optimizer.zero_grad()
            loss_e, loss_f, Ns = 0, 0, 0
            for i in batch:
                image = valid_data[i.item()]
                G = image['G'].to(device=device, dtype=dtype)
                dGdR = image['dGdR'].to(device=device, dtype=dtype)
                E = (image['E'] / image['N']).to(device=device, dtype=dtype)
                F = image['F'].to(device=device, dtype=dtype)
                Ns += image['N']

                G.requires_grad_(True)
                if p['ScaleMethod'] == 'Minmax':
                    sG = (G-scale['G_min']) / (scale['G_max']-scale['G_min'])
                    E = (E-scale['E_min']) / (scale['E_max']-scale['E_min'])
                    F = (F-scale['F_min']) / (scale['F_max']-scale['F_min'])
                elif p['ScaleMethod'] == 'Standard':
                    sG = (G - scale['G_mean']) / scale['G_std']
                    E = (E - scale['E_mean']) / scale['E_std']
                    F = (F - scale['F_mean']) / scale['F_std']

                E_pred = model(sG).sum()
                dEdG = torch.autograd.grad(E_pred, G, create_graph=True)[0]
                F_pred = - torch.einsum('ij, ijkw -> kw', dEdG, dGdR)

                loss_e += (E_pred - E)**2
                loss_f += ((F_pred - F)**2).sum()
                G.grad = None

            loss = Ce * loss_e/len(batch) + Cf * loss_f/(3*Ns)
            loss_valid += loss.item()

        loss_valid /= len(valid_dl)

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
            torch.save(model, p['WorkDir']+p['TaskName']+'/best_model.pt')
        if ep % 100 == 0:
            torch.save(model, p['WorkDir']+p['TaskName']+'/checkpoint.pt')
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

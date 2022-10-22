from easynn.descriptor import ACSF
from ase.io import read
import yaml
import glob
import torch
import numpy as np


class EASYDATA:
    """The class deals with data, such as transforming, loading, scaling.

    Args:
        params_file (str): yaml file path of parameters
    """

    def __init__(self, params_file):
        with open(params_file, 'r') as f:
            self.params = yaml.full_load(f)
        self.descriptor = ACSF(self.params)
        if self.params['Device'] == 'cpu':
            self.device = torch.device('cpu')
        elif self.params['Device'] == 'gpu':
            self.device = torch.device('cuda:0')

        if self.params['Dtype'] == 'float64':
            self.dtype = torch.float64
        elif self.params['Dtype'] == 'float32':
            self.dtype = torch.float32

    def transform_data(self):
        """Transform data.
        """

        p = self.params
        for k in p['Data'].keys():
            traj = read(p['Data'][k]['Path'], ':')
            ts = p['Data'][k]['TrainSize']
            train_images = traj[:ts]
            self.descriptor.transform(train_images, p['WorkDir']+f"data/train-{k}")
            vs = p['Data'][k]['ValidSize']
            valid_images = traj[ts:ts+vs]
            self.descriptor.transform(valid_images, p['WorkDir']+f"data/valid-{k}")

    def load_data(self, train=True):
        """Load data from data folders.

        Args:
            train (bool): whether to load training data
            size (int): number of data to load

        Returns:
            data (dict): data dictionary, including E, F, N, G, dGdR
        """

        p = self.params
        E, F, N = [], [], []
        G, dGdR, ID = [], [], []
        for k in p['Data'].keys():
            if train:
                datapath = p['WorkDir']+f"data/train-{k}"
                datasize = p['Data'][k]['TrainSize']
            else:
                datapath = p['WorkDir']+f"data/valid-{k}"
                datasize = p['Data'][k]['ValidSize']
            files = glob.glob(datapath+"/*.pt")
            if p['Shuffle']:
                ids = np.random.choice(len(files), datasize, replace=False)
            else:
                ids = np.arange(0, len(files), len(files)//datasize)
            ids = torch.from_numpy(ids)
            for i in ids:
                datum = torch.load(datapath+f"/{i:06d}.pt")
                E.append(datum['E'] / datum['N'])
                F.append(datum['F'])
                N.append(datum['N'])
                G.append(datum['G'])
                dGdR.append(datum['dGdR'])
            ID.append(ids)
        data = {'E': E, 'F': F, 'N': N, 'G': G, 'dGdR': dGdR, 'ID': ID}
        return data

    def scale_data(self, data):
        """Find scaling constants.

        Args:
            data (dict): The data to be scaled.

        Returns:
            scale (dict): scale constants dictionary
        """

        p = self.params
        scale = {}
        if p['Scale']['Method'] == 'Standard':
            if p['Scale']['E']:
                E = torch.tensor(data['E'])
                scale['E_mean'] = E.mean().type(self.dtype).to(self.device)
                scale['E_std'] = E.std().type(self.dtype).to(self.device)
            if p['Scale']['F']:
                F = torch.cat(data['F'])
                scale['F_mean'] = F.mean().type(self.dtype).to(self.device)
                scale['F_std'] = F.std().type(self.dtype).to(self.device)
            if p['Scale']['G']:
                G = torch.cat(data['G'])
                scale['G_mean'] = G.mean(dim=0).type(self.dtype).to(self.device)
                scale['G_std'] = G.std(dim=0).type(self.dtype).to(self.device)
        return scale

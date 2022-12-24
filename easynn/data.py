from ase.io import read
import torch
import numpy as np


class EasyData:
    """The class deals with data, such as transforming, loading, scaling.

    Args:
        params_file (str): yaml file path of parameters
    """

    def __init__(self, data_file, descriptor, device='cpu', dtype=torch.float32):
        self.datafile = data_file
        self.descriptor = descriptor
        self.device = device
        self.dtype = dtype

    def split(self, train_size, shuffle=False):
        """Split data into training and validation set.

        Args:
            train_size (int): number of training images
            shuffle (bool): shuffle the data or not
        """

        traj = read(self.datafile, ':')
        if shuffle:
            train_ids = np.random.choice(len(traj), train_size, replace=False)
            train_images = traj[train_ids]
            valid_ids = np.setdiff1d(np.arange(len(traj)), train_ids)
            valid_images = traj[valid_ids]
        else:
            train_images = traj[:train_size]
            valid_images = traj[train_size:]
        return train_images, valid_images

    def transform(self, images, save_path):
        """Transform coordinates of ase images to descriptors.
        """
        acsf = self.descriptor.transform(images)
        torch.save(acsf, save_path)

    def scale(self, data, method):
        """Find scaling constants. The energy is divided by the number of atoms.

        Args:
            data (list): list of dict

        Returns:
            scale (dict): scale constants dictionary
        """

        E, F, G = [], [], []
        for d in data:
            E += [d[i]['E']/d[i]['N'] for i in range(len(d))]
            F += [d[i]['F'] for i in range(len(d))]
            G += [d[i]['G'] for i in range(len(d))]
        E = torch.tensor(E)
        F = torch.cat(F)
        G = torch.cat(G)

        scale = {}
        if method == 'Minmax':
            scale['E_min'] = E.min().item()
            scale['E_max'] = E.max().item()
            scale['F_min'] = F.min().item()
            scale['F_max'] = F.max().item()
            scale['G_min'] = G.min().item()
            scale['G_max'] = G.max().item()
            scale['method'] = 'Minmax'
        elif method == 'Standard':
            scale['E_mean'] = E.mean().item()
            scale['E_std'] = E.std().item()
            scale['F_mean'] = F.mean().item()
            scale['F_std'] = F.std().item()
            scale['G_mean'] = G.mean(dim=0).type(self.dtype).to(self.device)
            scale['G_std'] = G.std(dim=0).type(self.dtype).to(self.device)
            scale['method'] = 'Standard'
        return scale

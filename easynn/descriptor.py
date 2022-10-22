import torch
from functorch import jacrev
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
import numpy as np
import os


def get_neighbors(image, r_c):
    """Obtain the information of coordinates and atomic numbers of neighbor
       atoms.

    Args:
        image (ase.Atoms): object of ase.Atoms
        r_c (float): cutoff radius [Angstrom]

    Returns:
        image_nbr_pos (list): coordinates of neighbor atoms
        image_nbr_sym (list): atomic numbers of neighbor atoms
    """

    cutoffs = r_c/2 * np.ones(len(image))
    nl = NeighborList(
        cutoffs.tolist(),
        skin=0.0,                    # calculate when update() is called
        self_interaction=False,      # doesn't count center atom
        bothways=True,               # count all neighbors
        primitive=NewPrimitiveNeighborList)
    nl.update(image)

    image_nbr_sym, image_nbr_pos = [], []
    for i in range(len(image)):  # atom loop
        ids, offsets = nl.get_neighbors(i)
        image_nbr_pos.append(
            # faster than np.dot
            image.positions[ids] + offsets @ image.get_cell())
        image_nbr_sym.append(image.numbers[ids])
    return image_nbr_pos, image_nbr_sym


def groupby(atom_nbr_sym, elements):
    """Seperate neighbor atoms of the centred atom according to their atomic
       numbers.

    Args:
        atom_nbr_sym (numpy.ndarray): atomic numbers of neighbor atoms
        elements (list): atomic numbers of elements

    Returns:
        nbr_groups (dict): dictionary of neighbor atom groups, keys are
                           'radial' and 'angular'.
    """

    atom_nbr_sym = torch.from_numpy(atom_nbr_sym)
    keys_1 = torch.tensor(elements)
    keys_2 = torch.combinations(keys_1, 2, with_replacement=True)

    nbr_groups = {'radial': {}, 'angular': {}}
    for k1 in keys_1:
        # nonzero return shape as [-1, 1], need to flatten
        nbr_groups['radial'][k1] = (atom_nbr_sym == k1).nonzero().flatten()

    ids = torch.arange(len(atom_nbr_sym))
    id_pairs = torch.combinations(ids, 2)
    sym_pairs = atom_nbr_sym[id_pairs].sort().values
    for k2 in keys_2:
        # nbr_groups['angular'][k2].shape = [2, n]
        nbr_groups['angular'][k2] = id_pairs[(sym_pairs == k2).all(1)].T
    return nbr_groups


class ACSF:
    """Descriptors used in Behler Parrinello neural network force field.

    Args:
        params_file (str): yaml file path of parameters

    """

    def __init__(self, params):
        self.params = params
        if self.params['Device'] == 'cpu':
            self.device = torch.device('cpu')
        elif self.params['Device'] == 'gpu':
            self.device = torch.device('cuda:0')

        if self.params['Dtype'] == 'float64':
            self.dtype = torch.float64
        elif self.params['Dtype'] == 'float32':
            self.dtype = torch.float32

        self.elements = self.params['Elements']
        self.r_c = self.params['CutoffRadius']
        self.acsf_params = self.acsf_params()

    def acsf_params(self):
        """Return the parameters of symmetry functions.

        Returns:
            params (dict): parameters of symmetry functions.
        """

        params = {}
        if self.params['Method']['name'] == 'Giulio Imbalzano':
            n = self.params['Method']['n']
            m = np.arange(n+1)

            eta_1 = (n**(m/n)/self.r_c)**2
            Rs_1 = np.zeros_like(eta_1)
            Rs_2 = self.r_c/(n**(m/n))
            Rs_n_m = np.flip(Rs_2)  # Rs_{n-m}
            eta_2 = 1/(Rs_n_m[:-1] - Rs_n_m[1:])**2
            eta = np.concatenate((eta_1, eta_2[1:]))
            Rs = np.concatenate((Rs_1, Rs_2[1:-1]))
            params['radial'] = {'eta': eta,
                                'Rs': Rs}

            eta = (n**(m/n)/self.r_c)**2
            ld = np.tile(np.array([1, -1]), 3*len(eta))
            zeta = np.tile(np.array([1, 4, 16]).repeat(2), len(eta))
            params['angular'] = {'eta': eta.repeat(6),
                                 'zeta': zeta,
                                 'lambda': ld}
        return params

    def radial(self, nbr_vec, groups):
        """Calculates radial symmetry functions."""

        eta = torch.from_numpy(self.acsf_params['radial']['eta'])
        eta = eta.to(self.device, self.dtype)
        Rs = torch.from_numpy(self.acsf_params['radial']['Rs'])
        Rs = Rs.to(self.device, self.dtype)

        G_rad = []
        for k, ids in groups.items():
            if len(ids) == 0:
                G_rad.append(torch.zeros_like(eta).to(self.device))
                continue
            vec = nbr_vec[groups[k].to(self.device, torch.long)]
            r = torch.norm(vec, dim=1, keepdim=True)
            G = torch.exp(-eta * (r - Rs)**2) * self.fcut(r)
            G_rad.append(G.sum(dim=0))
        return torch.cat(G_rad)

    def angular(self, nbr_vec, groups):
        """Calculates angular symmetry functions."""

        eta = torch.from_numpy(self.acsf_params['angular']['eta'])
        eta = eta.to(self.device, self.dtype)
        zeta = torch.from_numpy(self.acsf_params['angular']['zeta'])
        zeta = zeta.to(self.device, self.dtype)
        ld = torch.from_numpy(self.acsf_params['angular']['lambda'])
        ld = ld.to(self.device, self.dtype)

        G_ang = []
        for k, ids in groups.items():
            if len(ids) == 0:
                G_ang.append(torch.zeros_like(eta).to(self.device))
                continue
            vec = nbr_vec[groups[k].to(self.device, torch.long)]
            r_ij, r_ik = torch.norm(vec, dim=-1, keepdim=True)
            cos_jik = torch.sum(vec[0]*vec[1], dim=1, keepdim=True)/(r_ij*r_ik)
            G = 2 * (torch.exp(-eta * (r_ij**2 + r_ik**2))
                     * self.fcut(r_ij) * self.fcut(r_ik)
                     * (0.5001 + 0.5*ld*cos_jik) ** zeta)
            G_ang.append(G.sum(dim=0))
        return torch.cat(G_ang)

    def fcut(self, r):
        """Cosine cutoff function."""
        result = 0.5 * (torch.cos(torch.pi * r / self.r_c) + 1)
        return result

    def calculate(self, image_pos, image_nbr_pos, image_nbr_sym):
        """Calculate the symmetry functions for a single image.

        Args:
            image (ase.Atoms): object of ase.Atoms

        Returns:
            G (torch.Tensor): symmetry functions values
        """

        G = []
        for i in range(len(image_pos)):  # atom loop
            nbr_pos = torch.from_numpy(image_nbr_pos[i])
            nbr_pos = nbr_pos.to(self.device, self.dtype)
            nbr_vec = nbr_pos - image_pos[i]
            nbr_groups = groupby(image_nbr_sym[i], self.elements)
            G_rad = self.radial(nbr_vec, nbr_groups['radial'])
            G_ang = self.angular(nbr_vec, nbr_groups['angular'])
            G.append(torch.cat([G_rad, G_ang]))
        return torch.stack(G)

    def transform(self, images, data_path=None):
        """Calculate the symmetry functions for a list of images.

        Args:
            images (list): list of ase.Atoms [trajectory]

        Returns:
            acsf (torch.Tensor): symmetry functions
        """

        os.makedirs(data_path, exist_ok=True)
        for i, image in enumerate(images):
            image_nbr_pos, image_nbr_sym = get_neighbors(image, self.r_c)
            image_pos = torch.from_numpy(image.positions).type(self.dtype)
            image_pos = image_pos.to(self.device).requires_grad_()
            G = self.calculate(image_pos, image_nbr_pos, image_nbr_sym)
            dGdR = jacrev(self.calculate, argnums=0)(
                image_pos, image_nbr_pos, image_nbr_sym)
            e = torch.tensor(image.get_potential_energy())
            f = torch.from_numpy(image.get_forces(apply_constraint=False))
            torch.save({
                'E': e.type(self.dtype),
                'F': f.type(self.dtype),
                'N': len(image),
                'G': G.detach(),
                'dGdR': dGdR.detach(),
                'numbers': torch.from_numpy(image.numbers)
            }, f"{data_path}/{i:06d}.pt")

import ase.calculators.calculator as ase_calc
from easynn.descriptor import ACSF, get_neighbors
from functorch import jacrev
import torch
import yaml


class EasyCal(ase_calc.Calculator):
    """Link EasyNN to ASE.

    Args:
        params_file (str): Path to the parameters file.
        model_file (str): Path to the model file.
        scale_file (str): Path to the scale file.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, params_file, model_file, scale_file):
        super().__init__()
        with open(params_file, 'r') as f:
            self.p = yaml.full_load(f)
        self.descriptor = ACSF(self.p)
        self.model = torch.load(model_file)
        self.s = torch.load(scale_file)
        if self.p['Device'] == 'cpu':
            self.device = torch.device('cpu')
        elif self.p['Device'] == 'gpu':
            self.device = torch.device('cuda:0')

        if self.p['Dtype'] == 'float64':
            self.dtype = torch.float64
        elif self.p['Dtype'] == 'float32':
            self.dtype = torch.float32

    def calculate(self, image, properties=['energies'],
                  system_changes=ase_calc.all_changes):
        """Calculate the energy and forces.

        Args:
            image (ase.Atoms): The image to calculate.
            properties (list): The properties to calculate.
            system_changes (list): The system changes.

        """

        super().calculate(image, properties, system_changes)
        image_nbr_pos, image_nbr_sym = get_neighbors(image, self.p['CutoffRadius'])
        image_pos = torch.from_numpy(image.positions).type(self.dtype)
        image_pos = image_pos.to(self.device).requires_grad_()
        G = self.descriptor.calculate(image_pos, image_nbr_pos, image_nbr_sym)
        dGdR = jacrev(self.descriptor.calculate, argnums=0)(
            image_pos, image_nbr_pos, image_nbr_sym)
        self.model.eval()
        G = G.detach().cpu()
        dGdR = dGdR.detach().cpu()
        G.requires_grad_()
        G = (G - self.s['G_mean']) / self.s['G_std']
        E_pred = self.model(G).sum()
        self.results['energy'] = E_pred*self.s['E_std'] + self.s['E_mean']
        self.results['energy'] = self.results['energy'].detach().numpy()
        self.results['free_energy'] = self.results['energy']

        if 'forces' in properties:
            dEdG = torch.autograd.grad(E_pred, G, create_graph=True)[0]
            F_pred = - torch.einsum('ij, ijkw -> kw', dEdG, dGdR)
            self.results['forces'] = F_pred*self.s['F_std'] + self.s['F_mean']
            self.results['forces'] = self.results['forces'].detach().numpy()

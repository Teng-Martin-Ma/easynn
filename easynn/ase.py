import ase.calculators.calculator as ase_calc
from easynn.descriptor import ACSF, get_neighbors
from functorch import jacrev
import torch



class EasyCal(ase_calc.Calculator):
    """Link EasyNN to ASE.

    Args:
        params_file (str): Path to the parameters file.
        model_file (str): Path to the model file.
        scale_file (str): Path to the scale file.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, wd, device='cpu', dtype=torch.float32):
        super().__init__()
        if wd[-1] != '/':
            wd += '/'

        self.model = torch.load(wd+'best_model.pt')
        self.s = torch.load(wd+'scale.pt')
        self.dtype = dtype
        self.device = device
        p = torch.load(wd+'acsf.pt')
        self.r_c = p['r_c']
        self.descriptor = ACSF(wd, p['elements'], self.r_c, device, dtype,
                               params=p['acsf_params'])

    def calculate(self, image, properties=['energies'],
                  system_changes=ase_calc.all_changes):
        """Calculate the energy and forces.

        Args:
            image (ase.Atoms): The image to calculate.
            properties (list): The properties to calculate.
            system_changes (list): The system changes.

        """

        super().calculate(image, properties, system_changes)
        image_nbr_pos, image_nbr_sym = get_neighbors(image, self.r_c)
        image_pos = torch.from_numpy(image.positions).type(self.dtype)
        image_pos = image_pos.to(self.device).requires_grad_()
        G = self.descriptor.calculate(image_pos, image_nbr_pos, image_nbr_sym)
        dGdR = jacrev(self.descriptor.calculate, argnums=0)(
            image_pos, image_nbr_pos, image_nbr_sym)
        self.model.eval()
        G = G.detach().cpu()
        dGdR = dGdR.detach().cpu()

        G.requires_grad_()
        if self.s['method'] == 'Minmax':
            sG = (G-self.s['G_min'])/(self.s['G_max']-self.s['G_min'])
        elif self.s['method'] == 'Standard':
            sG = (G - self.s['G_mean']) / self.s['G_std']

        E_pred = self.model(sG).sum()
        if self.s['method'] == 'Minmax':
            self.results['energy'] = E_pred * \
                (self.s['E_max']-self.s['E_min']) + self.s['E_min']
        elif self.s['method'] == 'Standard':
            self.results['energy'] = E_pred*self.s['E_std'] + self.s['E_mean']
        self.results['energy'] = self.results['energy'].item()*len(image)
        self.results['free_energy'] = self.results['energy']

        if 'forces' in properties:
            dEdG = torch.autograd.grad(E_pred, G, create_graph=True)[0]
            F_pred = - torch.einsum('ij, ijkw -> kw', dEdG, dGdR)
            self.results['forces'] = F_pred*self.s['F_std'] + self.s['F_mean']
            self.results['forces'] = self.results['forces'].detach().numpy()

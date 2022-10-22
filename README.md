# EasyNN

EasyNN is a simple neural network library written in Python. It is designed to be easy to use and easy to extend. **EasyNN is under construction, which is therefore, not intended for production use.**

## Structure
easynn  
├── easynn  
│   ├── \_\_init\_\_.py  
│   ├── data.py  
│   ├── descriptor.py  
│   ├── model.py  
│   ├── train.py  
├── configs  
├── test  
├── README.md  

## Functionality

* Fast calulate Behler Parinello descriptors
* Train BPNN, currently only support 2nd generation BPNN
* Monitor training process with Wandb


## Dependencies

EasyNN requires the following Python packages:

* pytorch >= 1.12.0
* functorch >= 0.2.1
* ase >= 3.22.1
* matplotlib >= 3.5.2
* wandb >= 0.13.4
* tqdm >= 4.64.1

## Installation

Clone the repository and add into PYTHONPATH:

```bash
git clone https://github.com/Teng-Martin-Ma/easynn.git
export PYTHONPATH=$PYTHONPATH:/path/to/easynn
```

## Usage
```python
from easynn.train import train
train('configs/params.yaml')
```

## TODO
* [ ] Add more optimizers
* [ ] Add more activation functions
* [ ] Connect to EasyPLOT to analyze the results
* [ ] Log training process in config file and train function


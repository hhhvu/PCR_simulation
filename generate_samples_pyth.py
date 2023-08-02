import os
import argparse
import numpy as np
import pandas as pd
from decimal import *
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from resnet import EKGResNetModel
from base import fit_model
from torch.utils.data import DataLoader

def update_delta(gamma_j, beta):
    return 1/gamma_j * ((gamma_j*(beta - 1) + 1)**(1/(1 - beta)))

def update_eff(n_a, E_o, S_o, delta_e, k_C, n_d):
    # if E_o - n_a* S_o is positive, the exponential value of this expression will explode in calculation.
    # The original form of efficiency formula contains negative exponent and
    # the simplified form contains the positive exponent
    
    if (E_o - n_a*S_o) >= 0:
        return n_d * (n_a - \
                        ((E_o - n_a*S_o)*n_a*np.exp(-(E_o - n_a * S_o) * k_C * delta_e))/ \
                        (E_o - n_a*S_o*np.exp(-(E_o - n_a*S_o) * k_C * delta_e)))
    else:
        return n_d * (n_a - \
                    ((E_o - n_a*S_o) * n_a)/ \
                        (E_o * np.exp((E_o - n_a * S_o) * k_C * delta_e) - n_a * S_o))
    
def simulate_vals(D_init, E_init, P_init, delta_e, n_d, n_dE, k_C, beta, n_cycle=50):
    S_o, gamma, delta, n_a, eff = [], [], [], [], []
    P_o, E_o, D_e = [P_init], [E_init], [D_init]


    for i in range(n_cycle):
        S_o_j = n_d*D_e[i]
        S_o.append(S_o_j)

        gamma_j = S_o_j/P_o[i]
        gamma.append(gamma_j)

        delta_j = update_delta(gamma_j, beta=beta)
        delta.append(delta_j)

        n_a_j = 1/gamma_j - delta_j
        n_a.append(n_a_j)

        E_o_j = (n_dE**(i+1)) * E_o[0]
        E_o.append(E_o_j)

        eff_j = update_eff(n_a=n_a_j, E_o=E_o_j, S_o=S_o_j, delta_e=delta_e, k_C=k_C, n_d=n_d)
        eff.append(eff_j)
        # print('At cycle {}, PCR efficiency is {}'.format(i, eff_j))

        D_e_j = (eff_j + 1)*D_e[i]
        D_e.append(D_e_j)

        P_o_j = P_o[i] - eff_j * S_o_j
        P_o.append(P_o_j)
    
    return eff, n_a, D_e, S_o, E_o, P_o, gamma, delta

class SynthCurveDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, index):
        return torch.tensor(self.inputs[index]).unsqueeze(dim=0), torch.tensor(self.outputs[index])
    
    def __len__(self):
        return len(self.inputs)

# define constants
k_C = Decimal(15)
n_d = Decimal(1)
n_dE = Decimal(0.99)

beta = Decimal(23)

P_init = Decimal(9.e5)
E_init = Decimal(1e5)
delta_e = Decimal(50)

if __name__ == "__main__":

    #################################
    #Generate dataset
    #################################
    D_inits = []
    D_es = []

    for i in np.linspace(-6, 6, 10000): #np.linspace(0, 100000, 10000):
        D_e = simulate_vals(D_init=Decimal(10.0**i), #Decimal(i) 
                            E_init=E_init, 
                            P_init=P_init, 
                            delta_e=delta_e,
                            n_d=n_d,
                            n_dE=n_dE,
                            k_C=k_C,
                            beta=beta,
                            n_cycle=49)[2]

        # plt.plot(D_e, label = 'modelled curve')
        # plt.legend()
        # plt.grid()

        D_inits.append(np.array([10.0**i]))
        D_es.append(np.array([float(str(x)) for x in D_e]))

    curves_dict = {'D Init': D_inits, 'D Values': D_es}
    curves_df = pd.DataFrame(curves_dict)

    print(curves_df.head())
    # curves_df.to_csv('data/curves_vary_D_init_params.csv', index=False)

    #initialize dataset
    data_train, data_val, labels_train, labels_val = train_test_split(D_es, D_inits, test_size=0.2, random_state=42)
    train_dataset = SynthCurveDataset(data_train, labels_train)
    val_dataset = SynthCurveDataset(data_val, labels_val)

    print(len(train_dataset))
    print(len(val_dataset))
    print(train_dataset[0])
    print(val_dataset[0])

    NUM_EPOCHS = 20
    batch_size = 50

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    example_data, example_label = train_dataset[0]

    if len(example_data.shape) == 1:
        n_samples = example_data.shape[0]
        n_channels = 1
    else:
        n_channels, n_samples = example_data.shape

    n_outputs = example_label.shape[0]
    regress = [True]*n_outputs

    print("cuda available: " + str(torch.cuda.is_available()))

    model = EKGResNetModel(n_channels=n_channels, n_samples=n_samples, n_outputs=n_outputs, num_rep_blocks=8, kernel_size=16, regress=[True]) #original num_rep_blocks 32
    fit_model(model, train_dataloader, val_dataloader, save_path="output/resnet_lr1e-5", max_epochs=NUM_EPOCHS, learning_rate=1e-5)
import os
import pandas as pd
import numpy as np
from decimal import *
import pickle as pkl
import matplotlib.pyplot as plt

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
    
def simulate_vals(D_init, E_init, P_init, delta_e, n_d, n_dE, k_C, beta, n_cycle=45):
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

betas = [5, 10, 100, 1000]
D_inits = np.linspace(start=3e-3, stop=3e7, num=120)
P_inits = np.linspace(start=2e10, stop=6e11, num=30)
# E_inits = list(range(10000, 1300000, 15000)) # [Decimal(12.6e11), Decimal(6.3e11), Decimal(2.1e11)]
delta_e = Decimal(20) #[Decimal(5), Decimal
n_dEs = np.linspace(start=0.2, stop=0.9, num=7)
n_d = Decimal(1)
k_C = Decimal(15)

for i, P_init in enumerate(P_inits):
    print('Simulating for P_init = ', P_init)
    c, params, failed_params = [], [], []
    E_inits = np.linspace(0.79*P_init, 1.06*P_init, 10)
    for D_init in D_inits:
        for nde in n_dEs:
            for beta in betas:
                for E_init in E_inits:
                    D_init, P_init, nde, beta, E_init = Decimal(D_init), Decimal(P_init), Decimal(nde), Decimal(beta), Decimal(E_init)
                    try:
                        _, _, D_e, _, _, _, _, _ = simulate_vals(D_init=D_init,
                                                                E_init=E_init,
                                                                P_init=P_init,
                                                                delta_e=delta_e,
                                                                n_d=n_d,
                                                                n_dE=nde,
                                                                k_C=k_C,
                                                                beta=beta,
                                                                n_cycle=45)
                        c.append(D_e)
                        params.append([P_init, D_init, E_init, nde, beta])
                    except:
                        failed_params.append([D_init, E_init, P_init, nde, beta])

    with open(os.path.join('../data/','file_' + str(i) + '.pkl'), 'wb') as fb:
        pkl.dump({'curves': np.array(c),
                'params': np.array(params)}, fb)

    with open(os.path.join('../data/', 'failed_params.pkl'), 'wb') as fp:
        pkl.dump(np.array(failed_params), fp)

print('Done simulating')

                    
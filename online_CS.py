'''
Bayesian Online Compressed Sensing (2016)
Paulo V. Rossi & Yoshiyuki Kabashima
'''

from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from numpy.random import normal

from utils import DlnH, DDlnH, G, H, moments


def simulation(method='standard'):
    signal_length = 2000
    alpha_max = 20
    sigma_n_2 = 1e-1

    phi = prior()
    P = posterior(signal_length, phi)
    x0 = generate_signal(signal_length, phi)
    
    print('Simulation parameters:')
    print('N='+str(signal_length)+', sparsity='+str(phi.rho)+
          ', noise='+str(sigma_n_2)+', alpha_max='+str(alpha_max))
    print('Measurement model: '+method+'\n')

    number_of_measurements = alpha_max*signal_length
    mean_square_error = np.zeros(number_of_measurements)
    for measurement in range(number_of_measurements):
        P = update_posterior(P, phi, x0, signal_length, sigma_n_2, method)
        mean_square_error[measurement] = reconstruction_error(P, x0)
    plot_results(P, x0, mean_square_error, phi)


def prior():
    phi = namedtuple('prior_distribution', ['rho', 'sigma_x_2', 'bar_x'])
    phi.rho = 0.1
    phi.sigma_x_2 = 1.
    phi.bar_x = 0.
    return phi  
    
    
def posterior(signal_length, phi):
    P = namedtuple('posterior_distribution', ['m', 'v', 'a', 'h'])
    P.m = np.zeros(signal_length)
    P.v = phi.rho * phi.sigma_x_2 * np.ones(signal_length)
    P.a = np.zeros(signal_length)
    P.h = np.zeros(signal_length)
    return P
    
    
def generate_signal (signal_length, phi):
    x0 = np.zeros(signal_length)
    number_of_non_zero_components = int(np.ceil(signal_length*phi.rho))
    x0[:number_of_non_zero_components] = normal(loc=phi.bar_x,
                                                scale=np.sqrt(phi.sigma_x_2),
                                                size=number_of_non_zero_components)
    return x0


def update_posterior(P, phi, x0, signal_length, sigma_n_2, method):
    A_t = measurement_vector(signal_length)
    P.a, P.h = update_and_project(method, A_t, x0, sigma_n_2, P)
    P.m, P.v = moments(P, phi)
    return P

 
def measurement_vector(signal_length):
    A_t = normal(size=signal_length)
    return A_t/norm(A_t)


def update_and_project(method, A_t, x0, sigma_n_2, P):
    m, v, a, h = P.m, P.v, P.a, P.h
    u0 = np.dot(A_t, x0)
    if sigma_n_2 > 0:
        noise = normal(scale=np.sqrt(sigma_n_2))
    else:
        noise = 0
    y = u0 + noise
    Delta = np.dot(A_t, m)
    chi = np.dot(A_t**2, v)
    if method == 'standard':
        da, dh = update_and_project_std(y, Delta, chi, sigma_n_2, A_t, m)
    elif method == '1bit':
        da, dh = update_and_project_1bit(y, Delta, chi, sigma_n_2, A_t, m)
    else:
        raise ValueError('Measurement model not recognized. Please use "standard" or "1bit".')
    return a+da, h+dh


def update_and_project_std(y, Delta, chi, sigma_n_2, A_t, m):
    da = A_t**2 / (sigma_n_2 + chi)
    dh = (y-Delta)*A_t / (sigma_n_2 + chi) + da*m
    return da, dh
    
    
def update_and_project_1bit(y, Delta, chi, sigma_n_2, A_t, m):
    y = np.sign(y)
    u = y * np.dot(A_t, m)
    chi_prime = chi + sigma_n_2
    z = -u/np.sqrt(chi_prime)
    da = -A_t**2/chi_prime * DDlnH(z) 
    dh = -y*A_t/np.sqrt(chi_prime) * DlnH(z) + da*m
    return da, dh

 
def reconstruction_error(P, x0):
    return norm(x0 - P.m)**2 / norm(x0)**2


def plot_results(P, x0, mse_t, phi):
    plt.subplots(figsize=(10,20))
    plt.subplot(211)
    plt.plot(np.arange(len(mse_t))/float(len(P.m)), 10*np.log10(mse_t), color='k')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'mse (dB)')
    plt.subplot(212)
    plt.plot(P.m, color='k', lw = 0.7, label=r'$m$')
    plt.scatter(range(int(len(x0)*phi.rho)), x0[:int(len(x0)*phi.rho)], \
        marker='o', facecolors='none', edgecolors='r', lw=1.5, label=r'$x^0$')
    plt.xlim([0,len(P.m)])
    plt.xlabel(r'Vector Component')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    simulation(method='1bit')
    #simulation(method='standard')

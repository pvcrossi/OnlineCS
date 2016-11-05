import numpy as np
from scipy.special import erfc


def DlnH(x):
    eps = 1e-30
    return (-G(x)/(H(x) + eps))*(x < 10) - x*(x>=10)


def DDlnH(x):
    eps = 1e-30
    return (x*G(x)/(H(x) + eps) - DlnH(x)**2)*(x < 10) - (x>=10)


def G(x):
    return np.exp(-np.power(x,2)/2)/np.sqrt(2*np.pi)


def H(x):
    return erfc(x/np.sqrt(2))/2


def moments(P, phi):
    def compute_aux(a, h, m, phi):
        return phi.rho/ ((1 - phi.rho)*np.sqrt(phi.sigma_x_2*a + 1)* \
            np.exp(-(h + phi.bar_x/phi.sigma_x_2)**2/(2*(a + 1./phi.sigma_x_2)) \
            + 0.5*phi.bar_x**2/phi.sigma_x_2) + phi.rho)
    def compute_m(aux, a, h, phi):
        return aux*( (h + phi.bar_x/phi.sigma_x_2)/(a + 1./phi.sigma_x_2))
    def compute_v(aux, a, h, phi):
        return aux*(1./(a + 1./phi.sigma_x_2)) + \
            aux*( (h + phi.bar_x/phi.sigma_x_2)/(a + 1./phi.sigma_x_2))**2 - m**2

    aux =  compute_aux(P.a, P.h, P.m, phi)
    m = compute_m(aux, P.a, P.h, phi)
    v = compute_v(aux, P.a, P.h, phi)
    return m, v
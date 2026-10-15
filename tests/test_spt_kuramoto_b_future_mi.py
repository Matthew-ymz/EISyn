import numpy as np
from scipy.special import ive,i0e

from scripts.spt_kuramoto_b_future_mi import information_matrix


def test_fourier_pair_kl_against_wrapped_cauchy_identity():
    coefficients=np.zeros((33,9,9),dtype=complex); coefficients[0]=1
    coefficients[1:,0,1]=.4**np.arange(1,33)*np.exp(.7j*np.arange(1,33))
    actual,diagnostic=information_matrix(coefficients,32,2048)
    assert abs(actual[0,1]+np.log(1-.4**2))<1e-12
    assert diagnostic['minimum_relative_density']>0
    assert diagnostic['normalization_error']<1e-14


def test_vm_difference_fourier_factors_against_direct_convolution():
    a,b=1.7,2.3
    coefficients=np.zeros((33,9,9),dtype=complex); coefficients[0]=1
    k=np.arange(1,33)
    coefficients[1:,0,1]=(ive(k,a)/i0e(a))*(ive(k,b)/i0e(b))
    actual,_=information_matrix(coefficients,32,2048)
    angles=np.arange(2048)*2*np.pi/2048
    concentration=np.sqrt(a*a+b*b+2*a*b*np.cos(angles))
    relative=np.exp(concentration-a-b)*i0e(concentration)/(i0e(a)*i0e(b))
    expected=np.mean(relative*np.log(relative))
    assert abs(actual[0,1]-expected)<1e-12

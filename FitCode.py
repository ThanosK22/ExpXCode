#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:27:24 2024

@author: thanos
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


m = (11.2-4.4)/0.1
y1 = 4.4
x1 = 6.2 
yw = 4.63739919


def standdev(x):
    return(np.sqrt(np.mean(x)))

def linfit(x):
    return(m*x +y1 -m*x1 - yw)

def linfit2(x,a,b): 
    return(a*x + b)
    

countrateAg = [14.1,12.8,13.1,13.7,17.3]

countrateZr = [11.0,10.1,11.4,10.9,12.0,14.7]

countrateNb = [0.9, 0.7, 1.4, 1.2, 0.9, 1.5, 1.1, 1.5, 2.5]

countrateIn = [0.5, 0.1, 0.9, 0.2, 0.2, 0.3, 0.6, 1.9]

countrateMo = [1.5, 1.1, 1.4, 1.4, 1.1, 1.6, 1.7,  2.5]

countrates = [countrateAg, countrateZr, countrateNb, countrateIn, countrateMo]

thevalue = np.zeros(6)

# for i in range(len(countrates)): 
#     print(np.mean(countrates[i]),2.5*standdev(countrates[i]))
#     thevalue[i]=(np.mean(countrates[i])+2.5*standdev(countrates[i]))
    
# print(thevalue)


# print(sp.optimize.root(linfit,6.25))

thetas = [4.951, 6.925, 6.535, 4.436, 6.203]
dthetas = [0.06214611, 0.05373165, 0.05805872, 0.07126616, 0.05113079]

# thetas = (np.pi/180)*thetas

lambdas = [ 5.648*np.sin((np.pi/180)*x) for x in thetas]
dlambdas = np.zeros(len(lambdas))
dlambdasplot = np.zeros(len(lambdas))
for i in range(len(lambdas)):
    dlambdas[i] = np.sqrt( (np.sin((np.pi/180)*thetas[i]) * 0.022)**2 + (5.648 * np.cos((np.pi/180)*thetas[i]) * (np.pi/180)*dthetas[i])**2 )
    dlambdasplot[i] = np.abs(-(1/2)*(lambdas[i]**(-3/2))*dlambdas[i])
# print(lambdas,dlambdas)

Z = [47, 40, 41, 49, 42]

params = sp.optimize.curve_fit(linfit2, Z, 1/np.sqrt(lambdas),maxfev=1000)
[a,b] = params[0]
[aerr,berr] = params[1]

# sinthetaerr = np.cos((np.pi/180)*Theta[peaklist])*Theta[peaklist]            
        
    
    
print('Coefficients a,b: ','%.4f'%a,'%.4f'%b)
print('Error in coefficients: ',np.sqrt(np.diag(params[1])))

fitxlist = np.linspace(np.min(Z),np.max(Z),1000)
fitylist = a*fitxlist + b

plt.figure(figsize=(9,6))
plt.errorbar(Z, 1/np.sqrt(lambdas), yerr = dlambdasplot,  ecolor = 'k',elinewidth=0.7,fmt = 'o',ms=3,c='k',label = 'Recorded Data Points')
plt.grid(linestyle='--')
plt.plot(fitxlist,fitylist,label='Line of Best Fit')
plt.xlabel('Z')
plt.ylabel('$λ^{-1/2}$ $(Å^{-1/2})$')
plt.legend()
plt.xticks(np.linspace(40,49,10))
plt.title('Absorbsion Edge Wavelength as a Function of Atomic Number')
# plt.savefig('Rplot',dpi=300)
plt.show()

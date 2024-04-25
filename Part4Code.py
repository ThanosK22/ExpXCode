#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:26:22 2024

@author: thanos
"""

import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt 

def linfit(x,a,b):
    return(a*x + b)


# def linfit2(x):
#     return(m*x +y1 -m*x1 - yw)


tol = 0.1
ndev = 2.5

txts = ['X350.txt','X325.txt','X300.txt','X285.txt','X270.txt','X255.txt','X240.txt','X225.txt','X210.txt','X195.txt','X180.txt','X165.txt','X150.txt']

num = 12
k = txts[num]

tws = np.zeros(len(txts))

V = [35,32.5,30,28.5,27,25.5,24.0,22.5,21,19.5,18,16.5,15]

# anglelist = np.loadtxt(k,delimiter = '\t')[:,0]

# countrate = np.loadtxt(k,delimiter = '\t')[:,1]

# countrateloop = np.zeros(len(countrate))
# anglelistloop = np.zeros(len(countrate))

# countrateloop[0] = countrate[0]
# anglelistloop[0] = anglelist[0]

# for i in range(len(countrate)-1):

#         countrateloop[i+1] = countrate[i+1]
#         anglelistloop[i+1] = anglelist[i+1]
        
        
#         params = sp.optimize.curve_fit(linfit,anglelistloop ,countrateloop,maxfev=1000)
#         [a,b] = params[0]
#         [aerr,berr] = params[1]
        
#         if np.abs(a) >= tol: 
#             print(a)
#             countrateloop = countrateloop[:i]
#             anglelistloop = anglelistloop[:i]
#             lamb = np.mean(countrateloop)
#             stdev = np.sqrt(lamb)
#             yw = lamb + ndev*stdev
            
#             y1 = countrate[i]
#             y2 = countrate[i+1]
#             x1 = anglelist[i]
#             x2 = anglelist[i+1]
#             m = (y2-y1)/(x2-x1)
            
#             tw = sp.optimize.root(linfit2,anglelist[i]+0.5)
#             # print(lamb,yw)
#             print(lamb,yw,tw)
#             break
        
# fitxlist = np.linspace(np.min(anglelistloop),np.max(anglelistloop),1000)
# fitylist = a*fitxlist + b

tws = [3.445,3.740,4.077,4.286,4.482,4.778,4.956,5.417,5.768,6.370,6.775,7.313,8.217]
Vplot = [(1/x) for x in V]
lambdas = [ 5.648*np.sin((np.pi/180)*x) for x in tws]


params = sp.optimize.curve_fit(linfit,Vplot ,lambdas,maxfev=1000)
[a,b] = params[0]
[aerr,berr] = params[1]

fitxlist = np.linspace(np.min(Vplot),np.max(Vplot),1000)
fitylist = a*fitxlist + b

print(a,np.sqrt(np.diag(params[1])))
# h = (a* (1.602176634) * (1/(299792458)) *10**(5) )

h = ( a*10**(-10) ) * (1.602176634*10**(-19)) * (1/299792458) *10**(3)

herr = (1.602176634*10**(-19))*(1/(299792458)) * (0.1485368*10**(-10)) * 10**(3)
print(h,herr)

# plt.figure(figsize=(12,8))
# plt.plot(anglelist,countrate,'o--',lw=0.7,ms=2,label='Data',c='b')
# plt.plot(fitxlist,fitylist,label='Linear Fit Over the 0 Countrate Area',c='orange')
# plt.grid(linestyle='--')
# plt.legend()
# plt.xlabel('Angle ($^{\circ}$)')
# plt.ylabel('Countrate ($s^{-1}$)')
# plt.title('Countrate Plot Over the Low Angle Region')
# plt.show

plt.figure(figsize=(9,6))
plt.plot(fitxlist,fitylist,label='Line Of Best Fit')
plt.errorbar(Vplot,lambdas, xerr = [(0.01/x) for x in V], yerr = [np.sqrt( (np.sin( (np.pi/180)*t ) * 0.022)**2 + (5.648*np.cos( (np.pi/180)*t ) * 0.05*(np.pi/180) )**2 ) for t in tws ],c='k',fmt = 'o',elinewidth=0.7, ms = 3, label = 'Recorded Data')
plt.grid(linestyle='--')
plt.xlabel('Inverse of Voltage ($kV^{-1}$)')
plt.ylabel('Wavelength (Å)')
plt.legend()
plt.title('Plot of λ as a function of 1/V')
# plt.savefig('hplot',dpi=300)
plt.show()







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:35:25 2024

@author: thanos
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

kb = 0.6309 #Amstrong
ka = 0.7108
thetaerr = 0.05

def fitfunc(x,a,b):
    return((a*x) + b)

data = np.loadtxt('XDataSi4.txt', delimiter='	',unpack=True, skiprows=1)

Theta, Countrate = data[::]

peaklist = sp.signal.find_peaks(Countrate, prominence = 30)[0]
# peaklist = peaklist1[2:]
print(Theta[peaklist])


peakerr = [(5.9-5.6) , (6.7 - 6.3) , (17.7-17.4) , (20 - 19.7)  , (23.9-23.6) , (27.2-26.7)]
cerr = 0.5

plt.figure(figsize=(9,6))
plt.plot(Theta, Countrate,label = 'Recorded Data')
plt.errorbar(Theta[peaklist], Countrate[peaklist], xerr = peakerr, yerr = cerr,  ecolor = 'k',elinewidth=0.7,capsize=4,label = 'Identified Peaks',fmt = 'o',ms=3,c='k')
plt.xlim(2.5,30)
plt.ylim(0)
plt.legend()
plt.xlabel('Theta - Angle of Incidence onto Crystal (Deg)')
plt.ylabel('Countrate (1/s)')
plt.title('Detected countrate graphed against the angle of incidence')

plt.grid(linestyle='--')
# plt.savefig('SpectrucmSi',dpi=300)
plt.show()


peakbeta = np.array([peaklist[0],peaklist[2],peaklist[4]])
peakalpha = np.array([peaklist[1],peaklist[3],peaklist[5]])



sinthetabeta = np.sin((np.pi/180)*Theta[peakbeta])
sinthetaalpha = np.sin((np.pi/180)*Theta[peakalpha])

n1 = np.sqrt(3)
n3 = np.sqrt(27)
n4 = np.sqrt(3*16)

sintheta = np.sin((np.pi/180)*Theta[peaklist])
nlambda = [n1*kb,n1*ka,n3*kb,n3*ka,n4*kb,n4*ka]

nlambdabeta = [n1*kb, n3*kb, n4*kb]
nlambdaalpha = [n1*ka, n3*ka,n4*ka]

params = sp.optimize.curve_fit(fitfunc, sintheta, nlambda,maxfev=1000,p0=[0.2,-0.2])
[a,b] = params[0]
[aerr,berr] = params[1]

sinthetaerr = np.cos((np.pi/180)*Theta[peaklist])*thetaerr*(np.pi/180)
        
    
    
print('Coefficients a,b: ','%.3f'%a,'%.3f'%b)
print('Error in coefficients: ',np.sqrt(np.diag(params[1])))

fitxlist = np.linspace(sintheta[0],sintheta[-1],1000)
fitylist = a*fitxlist + b

plt.figure(figsize=(9,6))

# plt.scatter(sinthetabeta,nlambdabeta,c='b',s=13)
# plt.scatter(sinthetaalpha,nlambdaalpha,c='r',s=13)
# plt.errorbar(sintheta, nlambda, xerr = sinthetaerr, yerr =peakerr,  ecolor = 'k',elinewidth=0.7,fmt = '',ms=3,c='k')
plt.errorbar(sinthetabeta, nlambdabeta, xerr = [sinthetaerr[0],sinthetaerr[2],sinthetaerr[4]], yerr = [peakerr[0],peakerr[2],peakerr[4]],  ecolor = 'r',elinewidth=0.7,fmt = 'o',ms=3,c='r',label='Κβ')
plt.errorbar(sinthetaalpha, nlambdaalpha, xerr = [sinthetaerr[1],sinthetaerr[3],sinthetaerr[5]], yerr = [peakerr[1],peakerr[3],peakerr[5]],  ecolor = 'b',elinewidth=0.7,fmt = 'o',ms=3,c='b',label='Κα')

plt.plot(fitxlist,fitylist,c='k',label='Line of Best Fit')
plt.title('Linear Regression of Peak Wavelength vs Angle')
plt.legend()
plt.xlabel('sinθ ')
plt.ylabel('nλ (Å)')
plt.grid(linestyle='--')
# plt.savefig('SinelineSi',dpi=300)
plt.show()
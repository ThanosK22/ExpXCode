#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:20:34 2024

@author: ozgur
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

dy1 = dy2= dy= 0.5
dT2 = dT1 = 0.05
 

Ag = [17.3, 29.6, 4.9, 5]
Zr = [18.1, 26.6, 6.9, 7]
Nb = [2.5, 7.2, 6.6, 6.7]
In = [1.9, 3.6, 4.4, 4.5]
Mo = [4.4, 11.2, 6.2, 6.3]
Caplist = [Ag, Zr, Nb, In, Mo]
m = np.zeros(len(Caplist))
y = [23.62072184, 20.22855617, 4.15043856, 2.50371371, 4.63739919]
# dm =[]
tlist = [23.62072184, 20.22855617,  4.15043856,  2.50371371,  4.63739919]
dT = np.zeros(len(Caplist))
for i in range(len(Caplist)):
    y1 = Caplist[i][0]
    y2 = Caplist[i][1]
    T1 = Caplist[i][2]
    T2 = Caplist[i][3]
    
    dm = np.sqrt( (dy2/(T2-T1))**2 + (-dy1/(T2-T1))**2 + (-dT2*(y2 - y1)/(T2-T1)**2)**2 + ( dT1*(y2-y1)/(T2-T1)**2)**2 )
    
    # dm= np.sqrt(((dy2/T2 - T1)**2)+((-dy1/T2 - T1)**2)+((-dT2*(y2-y1)/(T2-T1)**2)**2)+((-dT1*(y2-y1)/(T2-T1)**2)**2))
    # print(dm)
    
# for i in range(len(Caplist)):
#     y1 = Caplist[i][0]
#     y2 = Caplist[i][1]
#     T1 = Caplist[i][2]
#     T2 = Caplist[i][3]
#     m[i] = ((y2-y1)/0.1)
#     # y1 = Caplist[i][0]
#     # y2 = Caplist[i][1]
#     # T1 = Caplist[i][2]
#     # T2 = Caplist[i][3]
    m = (y2-y1)/(T2-T1)
    dT[i] = np.sqrt( (dy/m)**2 + (-dy1/m)**2 + (dm*(y1-y[i])/(m**2))**2 + (-dT1)**2 )

    # dT[i] = np.sqrt(((dy/m[i])**2)+((-dy1/m[i])**2)+((dm[i]*(-y[i]+y1))/(m[i]**2)**2)+((-dT1)**2))
    print(dT)
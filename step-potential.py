# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:23:33 2023

@author: win
"""

import numpy as np
import matplotlib.pyplot as plt

def get_data(txt):
    """
    Get colum data seperately 
    primitive data type
    a, 1
    b, 2
    return x = [a, b]
           y = [1, 2]
    """
    x = []
    y = []
    f = open(txt)
    for line in f:
        line = line.strip('\n')
        line = line.split('\t')
        x.append(float(line[0]))
        y.append(float(line[1]))     
    f.close()

    return x , y



def cut_range(file,step,time_span):
    x, y = get_data(file)
    total_data = int(len(x)*0.01/time_span)
    single_list = int(len(x)/total_data)
    ocex = []
    ocey = []
    for i in range(total_data):
        singlex = []
        singley = []
        for j in range(single_list):
            singlex.append(x[i*(single_list)+j])
            singley.append(y[i*(single_list)+j])
        
        ocex.append(singlex)
        ocey.append(singley)
    Anodex = ocex[1::2]
    Anodey = ocey[1::2]
    Cathodex = ocex[::2]
    Cathodey = ocey[::2]
    
    
        
    return Anodex, Anodey, Cathodex, Cathodey


def integrate(x, y):
    integrate_number = []
    for i in range(len(y)):
        min_index = y[i].index(min(y[i]))
        new_y = y[i][:min_index] 
        new_y = [i - new_y[-1] for i in new_y]
        new_x = x[i][:min_index]
        assert len(new_x) == len(new_y)
        number = np.trapz(new_y, new_x)
        integrate_number.append(number)
    
    return integrate_number
        

def integrate_cathode(x, y):
    integrate_number = []
    for i in range(len(y)):
        min_index = y[i].index(max(y[i]))
        new_y = y[i][:min_index] 
        new_y = [i + new_y[-1] for i in new_y]
        new_x = x[i][:min_index]
        assert len(new_x) == len(new_y)
        number = np.trapz(new_y, new_x)
        integrate_number.append(number)
    
    return integrate_number

def get_capacitance(q,u):
    c = []
    for i in range(len(q)):
        c.append(q[i]/u[i])
    return c

from scipy.optimize import curve_fit

def cutting_cathode(x, y):
    ocen_y = []
    ocen_x = []
    for i in range(len(y)):
        min_index = 100
        new_y = y[i][:min_index] 
        new_y = [-(i + new_y[-1]) for i in new_y]
        new_x = x[0][:min_index]
        
        ocen_y.append(new_y)
        ocen_x.append(new_x)

    
    return ocen_x, ocen_y

def fitting_exp(x, y0, A, R0, B):
    x = np.array(x)
    return y0 + A*np.exp(R0*x) + B/np.sqrt(x)
       

def get_cdl(x,y,E):
    y0 = []
    A = []
    R0 = []
    B = []
    Rct = []
    Cdl = []

    for i in range(len(y)):
        x1 = x[0]
        y1 = y[i]
        popt, pcov = curve_fit(fitting_exp, x1, y1, [1, 1, 1, 1], maxfev=50000)
        y0=popt[0]       
        A = popt[1]
        Rs = (A/float(E[i]))
        Rct.append(Rs)
        R1 = popt[2]
        cdl = (-1/(R1*Rs))
        B = popt[3]
        Cdl.append(cdl)
        R0.append(R1)
    return Cdl, R0, Rct



def get_cdl1(x, y, E, Rs):
    
    constant = []
  
    K = []
    Cdl = []
    
    
    def fitting_cdl(t, A, Cdl, k, constant):
        
        t = np.array(t)
        return A*np.exp(-t/(Rs*Cdl)) + k/np.sqrt(t) + constant

    for i in range(len(y)):
        x1 = x
        y1 = y[i]
        popt, pcov = curve_fit(fitting_cdl, x1, y1, [1, 1, 1, 1], maxfev=1000000)
        A = popt[0]
        cdl=popt[1]       
        k = popt[2]

        c = popt[3]
        K.append(k)
        
        Cdl.append(cdl)
        constant.append(c)
        
        y_fit = fitting_cdl(x1, A, cdl, k, c)
        plt.plot(x1, y1)
        plt.plot(x1, y_fit, color='red', linewidth=1.0)
        plt.show()

    return Cdl
a, b, c, d = cut_range('NbSr-RuO2-PVC-21.txt', 0.01,10)
# h, k, l, m = cut_range('Nb-RuO2-PVC-11.txt', 0.01,10)

# fitx, fity = cutting_cathode(l, m)
e = integrate(a, b)
# f = integrate(h, k)
s = integrate_cathode(c, d)
# b = integrate_cathode(l, m)
x = np.arange(1.225, 1.645, 0.02)
print(len(x))
print(len(e))
c1 = get_capacitance(e, x)
# c2 = get_capacitance(f, x)
import pandas as pd

data_path = './Nb-RuO2.csv'
data = pd.DataFrame({'potential':x,"anode_int":e, 'cathode':s})
data.to_csv('NbSr-RuO2', index=False, sep=',')

# print(fity)

# x1 =  np.arange(1.245, 1.645, 0.02).tolist()
# a1 = get_cdl1(fitx[0], fity[1:25], x1[1:25], 15)
# cathode_cdl = plt.scatter(fitx[20], fity[20])
# print(a1)
# a1 = get_cdl1(fitx[20], fity[20], x1[20], 15)
# print(a1)
# print(c1)
# print(c2)
s1=plt.scatter(x, e, label='Ni-anode')
s2=plt.scatter(x, s, label='Ni-caanode')

# s2=plt.scatter(x, f, c="r", label='NiAu-anode')

# cathode_cdl = plt.scatter(x1[1:25], a1)

# s1= plt.scatter(x1, s, label='Ni-cathode')
# s2= plt.scatter(x1, b, label='NiAu-cathode')
# s1=plt.scatter(x, c1, label='Ni-anode')
# s2=plt.scatter(x, c2, c="r", label='NiAu-anode')
# plt.legend((s1,s2),('Ni-anode','NiAu-anode'),loc='best')
### fitting
# x = fitx[20]
# y = fity[20]
# popt, pcov = curve_fit(fitting_exp, x, y, [1, 1, 1, 1], maxfev=50000)
# y0 = popt[0]
# A = popt[1]
# R0 = popt[2]
# B = popt[3]
# y_fit = fitting_exp(x, y0, A, R0, B)
# plt.plot(x, y)
# plt.plot(x, y_fit, color='red', linewidth=1.0)

plt.show()
# plt.show()
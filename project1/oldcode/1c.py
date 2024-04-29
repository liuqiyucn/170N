import numpy as np
import matplotlib.pyplot as plt
import os
import sympy
from pylab import *
import matplotlib.animation as animation
from IPython import display

x_plot = []
y_plot = []

def file_process(s):
    f = open(s, "r")
    lines = f.readlines()
    f.close()
    convert = []
    result = []

    for line in lines:
        for element in line.split():
            convert.append(float(element))
        result.append(convert)
        convert = []

    result = np.array(result)
    return result

result1 = file_process("result1_l.txt")
result2 = file_process("result2_l.txt")
result3 = file_process("result3_l.txt")

time = result1[:,0]
x = result1[:,2]
v = result1[:,3]



figure(1)
# animation
for i in range(x.size):

    x_plot.append(x[i])
    y_plot.append(v[i])
 
    # Mention x and y limits to define their range
    plt.xlim(-5,25)
    plt.ylim(-5,5)
    plt.title("(1,0) initial value phase plot")
    plt.xlabel("x")
    plt.ylabel("v")
     
    # Plotting graph
    plt.plot(x_plot, y_plot, color = 'green')
    plt.pause(0.01)

plt.cla()

figure(2)
time = result2[:,0]
x = result2[:,2]
v = result2[:,3]


x_plot = x_plot.clear()
y_plot = y_plot.clear()
x_plot = []
y_plot = []

#animation
for i in range(x.size):

    x_plot.append(x[i])
    y_plot.append(v[i])
 
    # Mention x and y limits to define their range
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.title("(2,0) initial value phase plot")
    plt.xlabel("x")
    plt.ylabel("v")
     
    # Plotting graph
    plt.plot(x_plot, y_plot, color = 'red')
    plt.pause(0.01)

plt.cla()
figure(3)
time = result3[:,0]
x = result3[:,2]
v = result3[:,3]


x_plot = x_plot.clear()
y_plot = y_plot.clear()
x_plot = []
y_plot = []


for i in range(x.size):

    x_plot.append(x[i])
    y_plot.append(v[i])
 
    # Mention x and y limits to define their range
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.title("(0,3) initial value phase plot")
    plt.xlabel("x")
    plt.ylabel("v")
     
    # Plotting graph
    plt.plot(x_plot, y_plot, color = 'blue')
    plt.pause(0.01)

plt.show() 


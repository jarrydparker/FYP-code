# Animtion code
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes2D
from matplotlib import animation
%matplotlib qt

import json
with open('normv_y.txt') as f:
    normv_y = json.load(f)
    f.close()
with open('normv_x.txt') as f:
    normv_x = json.load(f)
    f.close()
with open('posx25.txt') as f:
    posx = json.load(f)
    f.close()
with open('posy25.txt') as f:
    posy = json.load(f)
    f.close()
with open('Cint.txt') as f:
    C_int = json.load(f)
    f.close()
    
# Defining constants
n_boids= 500
time_steps = 1000
snapshot = 50
pos = np.zeros((snapshot,n_boids,2))
norm_v = []
n_c = 25
# finding J values from data
C_avg = 0.996885

def animate_func(num):
    # plt.clear() # clears figure
    # plt.figure(1)
    plt.clf()
    plt.plot(posx[num],posy[num],linestyle = 'none',c='blue',marker='.') # update boid locations
    
    # Setting Axes Limits
    plt.xlim([0, 2000]) # set axis limits to be constant
    plt.ylim([0, 2000])
    
    # Adding Figure Labels
    plt.title('500 boids \nTimestep = ' + str(num) + ' sec')
    plt.xlabel('x')
    plt.ylabel('y')
    

# Plotting the Animation
fig = plt.figure()
ax = plt.axes()
ani = animation.FuncAnimation(fig, animate_func, interval=100, frames=snapshot)
plt.show()

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from numba import jit, njit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from leapfrog import leapfrog
import os
import sys
warnings.filterwarnings('ignore')

# Constants
R_min = 25 # in kpc
M = 1 # in solar masses
G = 4491.9 # kpc^3 / (M * T^2) where  solar masses and T = 10^8 years
e_soft = .03  # softing parameter
e = 0.6
m1 = 1
m2 = 1
rmin = 25
dt = 0.005 # .1T
T = 25 # total time in 25*T where T=10e8 years 
step = int(T/dt) # total number of steps, should be INT
L_scale = 1 # kpc

def initialize_points(R_min):
    R = [] # python list that holds the distance of each ring of stars
    T = [] # python list that holds the number of stars in each ring
    for i in range(11):
        T.append(12 + 3 * i)
        R.append((.2 + i * (.05)) * R_min)
    return R, T

def rtpairs_np(r, n):
    out = np.empty((297, 4))
    idx = 0
    for i in range(len(r)):
        for j in range(n[i]):
            out[idx, 0] = 1e-35
            out[idx, 1] = r[i] * np.cos(j * (2 * np.pi / n[i]))
            out[idx, 2] = r[i] * np.sin(j * (2 * np.pi / n[i]))
            out[idx, 3] = 0
            idx += 1
    return out

def v_0(x, y, M):
    """Velocity Function in 2d
    r^2 = x^2 + y^2 from mass M
    """
    r = np.sqrt(x**2 + y ** 2)
    v = np.sqrt(G * M * r / (r ** 2 + e_soft ** 2))
    theta = np.arccos(x / r)
    vy = v * np.cos(theta)
    theta = np.arcsin(y / r)
    vx = -v * np.sin(theta)
    return vx, vy

def initialize_velocity(initial):
    vx, vy = v_0(initial[:, 1], initial[:, 2], M)
    vx = vx.reshape(-1, 1)
    vy = vy.reshape(-1, 1)
    vz = np.zeros(297).reshape(-1, 1)
    initial = np.hstack((initial, vx, vy, vz))
    return initial

def rot(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    a = 1 - np.cos(theta)
    rot_mat = np.array([[a + c, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
    return rot_mat

def rotate_galaxy(initial, angle):
    rot_mat = rot(np.radians(angle))
    galaxy = initial.copy()
    galaxy[:, 1:4] = (rot_mat @ galaxy[:, 1:4].T).T
    galaxy[:, 4:7] = (rot_mat @ galaxy[:, 4:7].T).T
    return galaxy

def orbit(phi, rmin, e, m1, m2, G):
    r = rmin * (1 + e) / (1 - e * np.cos(phi))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    vr = -e * np.sqrt(G * (m1 + m2) / (rmin * (1 + e))) * np.sin(phi)
    vphi = np.sqrt(G * (m1 + m2) / (rmin * (1 + e))) * (1 - e * np.cos(phi))
    vx = vr * np.sin(phi) - vphi * np.sin(phi)
    vy = vr * np.sin(phi) + vphi * np.cos(phi)
    return x, y, vx, vy

def center_mass_orbit():
    phi = np.linspace(0, 2 * np.pi, 100)
    x1, y1, vx1, vy1 = orbit(phi, rmin, e, m1, m2, G)
    x2, y2, vx2, vy2 = orbit(phi, rmin, e, m1, m2, G)

    # Center of mass adjustment
    x1, y1 = x1 / 2, y1 / 2
    x2, y2 = -x2 / 2, -y2 / 2
    vx1, vy1 = vx1 / 2, vy1 / 2
    vx2, vy2 = -vx2 / 2, -vy2 / 2

    return x1, y1, vx1, vy1, x2, y2, vx2, vy2

def update_galaxy_positions(ring_A, ring_B, xA, yA, vxA, vyA, xB, yB, vxB, vyB):
    ring_A[:, 1] += xA
    ring_A[:, 2] += yA
    ring_A[:, 4] += vxA
    ring_A[:, 5] += vyA

    ring_B[:, 1] += xB
    ring_B[:, 2] += yB
    ring_B[:, 4] += vxB
    ring_B[:, 5] += vyB

    return ring_A, ring_B

def initialize_system():
    R, T = initialize_points(R_min)
    initial = rtpairs_np(R, T)
    initial = initialize_velocity(initial)
    ring_A = rotate_galaxy(initial, -15)
    ring_B = rotate_galaxy(initial, -60)

    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = center_mass_orbit()
    t = 0
    vxA, vyA = vx2[t], vy2[t]
    vxB, vyB = vx1[t], vy1[t]
    xA, yA = x2[t], y2[t]
    xB, yB = x1[t], y1[t]

    center = np.zeros((2, 7))
    center[0] = np.array([1, xA, yA, 0, vxA, vyA, 0])
    center[1] = np.array([1, xB, yB, 0, vxB, vyB, 0])

    ring_A, ring_B = update_galaxy_positions(ring_A, ring_B, xA, yA, vxA, vyA, xB, yB, vxB, vyB)
    system = np.vstack((ring_A, ring_B))

    return center, system

def simulate(center, system, step, dt, G, M):
    out_star = [] 
    out_satellite = []
    out_star_v = []
    out_satellite_v = []

    position_star = center[:, 1:4]
    velocity_star = center[:, 4:7]
    position_satellite = system[:, 1:4]
    velocity_satellite = system[:, 4:7]

    out_star.append(position_star)
    out_satellite.append(position_satellite)
    out_star_v.append(velocity_star)
    out_satellite_v.append(velocity_satellite)

    for i in range(step):
        position_star, velocity_star, position_satellite, velocity_satellite = leapfrog(
            position_star, velocity_star, position_satellite, velocity_satellite, dt=dt, G=G, M=M
        )
        out_star.append(position_star)
        out_satellite.append(position_satellite)
        out_star_v.append(velocity_star)
        out_satellite_v.append(velocity_satellite)

    out_star = np.array(out_star)
    out_satellite = np.array(out_satellite)
    out_star_v = np.array(out_star_v)
    out_satellite_v = np.array(out_satellite_v)

    return out_star, out_satellite, out_star_v, out_satellite_v

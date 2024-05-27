from numba import jit, njit
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import random
plt.style.use(['science', 'grid', 'notebook'])

def weighted_die(num_steps):
    out = []
    p = np.array([3,3,1,1,1,1])
    p = p / np.sum(p)

    # generate a uniform random between 0 and 1
    # if it is less than 0.5, move left, else move right

    curr = np.random.choice([0,1,2,3,4,5])
    curr_prob = p[curr]

    for i in range(num_steps):
        if np.random.rand() < 0.5:
            nex = curr - 1 if curr > 0 else 5
        else:
            nex = curr + 1 if curr < 5 else 0
        
        nex_prob = p[nex]

        # p(x*)/p(x^n)
        acceptance = nex_prob / curr_prob

        if np.random.rand() <= acceptance:
            curr = nex
            curr_prob = nex_prob

        out.append(curr)
    
    out = [value + 1 for value in out]

    return out

def weighted_die_direct_sampling(num_steps):
    sides = [1, 2, 3, 4, 5, 6]
    weights = [3, 3, 1, 1, 1, 1]  # The weight of each die side
    
    results = []
    for _ in range(num_steps):
        # Selects a side based on the provided weights
        roll = random.choices(sides, weights=weights, k=1)[0]
        results.append(roll)
    
    return results

@jit(nopython=True)
def two_dim_ising(L : int, temp : float, num_steps : int):
    Lattice_matrix = np.ones((L, L), dtype=np.int32)

    # Energy definition
    E_arr = np.empty(num_steps + 1) # store energy at each step
    E_arr[0] = -2 * L**2 # initial energy

    # Spin definition
    S_arr = np.empty(num_steps + 1)
    S_arr[0] = L**2

    for step in range(num_steps):
        i = np.random.randint(L)
        j = np.random.randint(L)
        H = 0  # external magnetic field

        s_i = Lattice_matrix[i, j]
        s_top = Lattice_matrix[(i - 1) % L, j]
        s_bot = Lattice_matrix[(i + 1) % L, j]
        s_left = Lattice_matrix[i, (j - 1) % L]
        s_right = Lattice_matrix[i, (j + 1) % L]
        dE = 2 * s_i * (s_top + s_bot + s_left + s_right + H)

        # if we accept the change
        if dE <= 0 or np.random.rand() < np.exp(-dE/temp):
            Lattice_matrix[i, j] = -s_i
            E_arr[step+1] = E_arr[step] - dE # energy update
            S_arr[step+1] = S_arr[step] - 2*s_i # spin update     
        # if we dont accept the change
        else:
            E_arr[step+1] = E_arr[step] # energy update
            S_arr[step+1] = S_arr[step] # spin update
    
    return Lattice_matrix, E_arr, S_arr

def U(E_arr, num_steps, L):
    burn_in = num_steps // 10 # keep one percent as burn in
    E_eff = E_arr[burn_in:]
    return np.mean(E_eff)/(L**2)

def M(S_arr, num_steps, L):
    burn_in = num_steps // 10 # keep one percent as burn in
    S_eff = S_arr[burn_in:]
    return np.mean(S_eff)/L**2

def U_series(E_arr, num_steps, L):
    out = []
    for i in range(1, num_steps+1):
        val = np.mean(E_arr[0:i])
        out.append(val)
    out = np.array(out) / L**2
    return out

def M_series(S_arr, num_steps, L):
    out = []
    for i in range(1, num_steps+1):
        val = np.mean(S_arr[0:i])
        out.append(val)
    out = np.array(out) / L**2
    return out

def Magnetization(L_arr):
    for L_val in L_arr:
        T = np.linspace(0.1, 2.25, 100)
        M_arr = np.empty(T.size)
        for i in range(T.size):
            num_steps = int(1e7)
            arr, E, S = two_dim_ising(L = L_val, temp = T[i], num_steps = num_steps)
            M_arr[i] = M(S_arr=S, num_steps=num_steps, L = L_val)
        M_theory = (1-( np.sinh(2/T) )**-4 )**(1/8)
        plt.plot(T, M_arr, 'o', label = 'Simulation')
        plt.plot(T, M_theory, label = 'theory')
        plt.xlabel('T')
        plt.ylabel('Magnetization')
        plt.title(f'M(T) below critical temperature L = {L_val}')
        plt.legend()
        plt.show()
        T = np.linspace(2.7, 10, 100)
        M_arr = np.empty(T.size)
        for i in range(T.size):
            num_steps = int(1e7)
            arr, E, S = two_dim_ising(L = L_val, temp = T[i], num_steps = num_steps)
            # print(U(E_arr=E, num_steps=num_steps, L = L))
            M_arr[i] = M(S_arr=S, num_steps=num_steps, L = L_val)
        M_theory = np.zeros(T.size)
        plt.plot(T, M_arr, 'o', label = 'Simulation')
        plt.plot(T, M_theory, label = 'theory')
        plt.xlabel('T')
        plt.ylabel('Magnetization')
        plt.title(f'M(T) above critical temperature L = {L_val}')
        plt.legend()
        plt.show()

def configuration(Lattice, temp):
    # Create the plot using a binary color map: +1 (black), -1 (white)
    fig, ax = plt.subplots(dpi = 150)
    cmap = plt.cm.gray  # Colormap where high values are black, low values are white
    ax.imshow(Lattice, cmap=cmap, interpolation='none', vmin=-1, vmax=1)

    # Hide the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Spin configuration for T = {temp}')

    plt.show()

def plot_configuration(temp_arr):
    for temp in temp_arr:
        arr, E, S = two_dim_ising(L = 256, temp = temp, num_steps=int(1e8))
        configuration(arr, temp)
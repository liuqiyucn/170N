import numpy as np
from numba import jit

@jit(nopython=True)
def accel_majorStars(pos: np.ndarray, G : float, M : float):
    n = 2 # number of major mass body
    A_pos = pos[0]
    B_pos = pos[1]
    a = np.zeros((n, 3)) # acceleration array

    r_vec = A_pos - B_pos 
    r_mag = np.linalg.norm(r_vec) 
    acc = -(G*M*r_vec/(r_mag**2+2**2)**(3/2))
    A_acc = acc # accel of mass body A
    B_acc = -acc # accel of mass body B

    a[0] = A_acc 
    a[1] = B_acc
    return a 

@jit(nopython=True)
def accel_satellites(pos_satellites: np.ndarray, pos_majorStars: np.ndarray, G: float, M: float):
    n = pos_majorStars[:,0].size # number of major masses
    m = pos_satellites[:,0].size # number of satellites
    a = np.zeros((m, 3))

    for i in range(n):
        for j in range(m):
            r_vec = pos_satellites[j] - pos_majorStars[i] 
            r_mag = np.linalg.norm(r_vec) 
            a[j] = a[j] + (-G*M*r_vec/(r_mag**2+2**2)**(3/2))
    return a

@jit(nopython=True) 
def leapfrog(r_star: np.ndarray, v_star: np.ndarray, r_satellites: np.ndarray, v_satellites: np.ndarray, dt: float, G: float, M: float):

    # major stars leapfrog
    v_star = v_star+ 0.5*dt*accel_majorStars(r_star, G, M)
    r_star = r_star+ v_star*dt
    v_star = v_star+ 0.5*dt*accel_majorStars(r_star, G, M)

    # satellites leapfrog
    v_satellites = v_satellites + 0.5*dt*accel_satellites(r_satellites, r_star, G, M)
    r_satellites = r_satellites + v_satellites*dt
    v_satellites = v_satellites + 0.5*dt*accel_satellites(r_satellites, r_star, G, M)

    return r_star, v_star,r_satellites, v_satellites

def angular_momentum(out_star, out_satellite, out_star_v, out_satellite_v, mass_star, mass_satellite):
    """
    Calculate the total angular momentum of the system at each time step.

    Parameters:
    out_star (np.ndarray): Positions of the major stars over time (time, n, 3)
    out_satellite (np.ndarray): Positions of the satellites over time (time, m, 3)
    out_star_v (np.ndarray): Velocities of the major stars over time (time, n, 3)
    out_satellite_v (np.ndarray): Velocities of the satellites over time (time, m, 3)
    mass_star (float): Mass of the major stars
    mass_satellite (float): Mass of the satellites

    Returns:
    np.ndarray: Total angular momentum of the system at each time step (time, 3)
    """
    num_steps = out_star.shape[0]
    L_total = np.zeros((num_steps, 1))

    for t in range(num_steps):
        pos_star = out_star[t]
        vel_star = out_star_v[t]
        pos_satellites = out_satellite[t]
        vel_satellites = out_satellite_v[t]

        # Calculate angular momentum for major stars
        L_star = np.sum(np.cross(pos_star, mass_star * vel_star), axis=0)
        L_star = np.sum(L_star)

        # Calculate angular momentum for satellites
        L_satellites = np.sum(np.cross(pos_satellites, mass_satellite * vel_satellites), axis=0)
        L_satellites = np.sum(L_satellites)

        # Total angular momentum
        L_total[t] = L_star + L_satellites

    return L_total

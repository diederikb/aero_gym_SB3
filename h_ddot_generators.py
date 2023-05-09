#h_ddot_generators.py>

from aero_gym.envs.wagner import WagnerEnv
import numpy as np
import math

def impulse_profile(t, start_index, amplitude):
    h_ddot = np.zeros(len(t))
    h_ddot[start_index] = amplitude
    return h_ddot

def step_profile(t, start_index, amplitude):
    h_ddot = np.zeros(len(t))
    h_ddot[start_index:] = amplitude
    return h_ddot

def dstep_profile(t, delta_t, start_index, amplitude):
    h_ddot = np.zeros(len(t))
    h_ddot[start_index] = amplitude / delta_t
    return h_ddot

def dramp_profile(t, delta_t, start_index, amplitude):
    h_ddot = np.zeros(len(t))
    h_ddot[start_index:] = amplitude / delta_t
    return h_ddot

def random_steps_ramps(env: WagnerEnv):
    N_events_max = 20
    amplitude_max = 1.0
    
    t = np.linspace(0, env.t_max, int(env.t_max / env.delta_t) + 1)
    N_events = env.np_random.integers(N_events_max, endpoint=True)
    
    h_ddot = np.zeros(len(t))
    
    for _ in range(0,N_events):
        start_index = env.np_random.integers(len(t), endpoint=False)
        amplitude = 2 * amplitude_max * env.np_random.random() - amplitude_max
        event = env.np_random.integers(2, endpoint=False)
        if event == 0:
            h_ddot_event = step_profile(t, start_index, amplitude)
        else:
            h_ddot_event = impulse_profile(t, start_index, amplitude)
        
        h_ddot += h_ddot_event
        
    return h_ddot

def random_fourier_series(env: WagnerEnv, T=30, N_max=100):
    t = np.linspace(0, env.t_max, int(env.t_max / env.delta_t) + 1)
    
    N = env.np_random.integers(1, high=N_max)
    A = env.np_random.normal(0, 1, N)

    s = 0.0
    s += sum(np.sin(2*math.pi/T*n*t)*A[n]/(n+1) for n in range(0, N))
    return s

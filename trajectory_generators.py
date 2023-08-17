#trajectory_generators.py>

import numpy as np
import math

def impulse_profile(t, start_index, value):
    ar = np.zeros(len(t))
    ar[start_index] = value
    return ar

def step_profile(t, start_index, step):
    ar = np.zeros(len(t))
    ar[start_index:] = step
    return ar

"""
Create a profile of a ramp that increments every element with `increment` compared to the previous element starting at `start_index` + 1 and ending at `end_index`
"""
def ramp_profile(t, start_index, end_index, increment):
    ar = np.zeros(len(t))
    for i in range(start_index + 1, end_index + 1):
        ar[i:] += increment
    return ar

def slope_profile(t, start_index, increment):
    ar = np.zeros(len(t))
    for i in range(start_index + 1, len(t) + 1):
        ar[i] = ar[i-1] + increment
    return ar

def dstep_profile(t, delta_t, start_index, amplitude):
    ar = np.zeros(len(t))
    ar[start_index] = amplitude / delta_t
    return ar

def dramp_profile(t, delta_t, start_index, amplitude):
    ar = np.zeros(len(t))
    ar[start_index:] = amplitude / delta_t
    return ar

"""
Create a function that takes an AeroGym environment and generates a numpy array filled with `value` with a size corresponding to the maximum number of timesteps in the passed environment.
"""
def constant(value):
    def constant_array_generator(env):
        generated_array = np.full(int(env.t_max / env.delta_t) + 1, value)
        return generated_array
    return constant_array_generator

"""
Create a function that takes an AeroGym environment and generates a numpy array filled with random value between `min_value` and `max_value` with a size corresponding to the maximum number of timesteps in the passed environment.
"""
def random_constant(min_value, max_value):
    def random_constant_array_generator(env):
        value = env.np_random.uniform(low=min_value, high=max_value)
        generated_array = np.full(int(env.t_max / env.delta_t) + 1, value)
        return generated_array
    return random_constant_array_generator

"""
Create a function that takes an AeroGym environment and generates a numpy array filled with `value` with a size corresponding to the maximum number of timesteps in the passed environment.
"""
def impulse(value, t_impulse=0.0):
    def impulse_generator(env):
        idx_impulse = int(np.ceil(t_impulse / env.delta_t))
        generated_array = np.zeros(int(env.t_max / env.delta_t) + 1)
        generated_array[idx_impulse] = value
        return generated_array
    return impulse_generator

def random_mixed_events(*args):
    def random_mixed_events_generator(env):
        event_type = env.np_random.integers(len(args), endpoint=False)
        selected_event_generator = args[event_type]
        return selected_event_generator(env)
    return random_mixed_events_generator

"""
Create a function that takes an AeroGym environment and generates a numpy array with a size corresponding to the maximum number of timesteps in the passed environment and containing the derivative of a random trajectory of at most `n_events_max` ramps. The derivative has a maximum amplitude of `max_d_amplitude` and its integral has a maximum amplitude of `max_int_amplitude`. 
"""
def random_d_ramps(n_events_max=20, max_int_amplitude=1.0, max_d_amplitude=1.0):
    def random_d_ramps_generator(env):
    
        t = np.linspace(0, env.t_max, int(env.t_max / env.delta_t) + 1)

        # compute max int event amplitudes to avoid exceeding max_d_amplitude
        max_slope = max_d_amplitude * env.delta_t

        # generate list of at most n_event_max list indices where the events take place
        n_events = env.np_random.integers(0, high=n_events_max, endpoint=True)
        start_index_list = np.sort(env.np_random.integers(0, high=len(t)-2, size=n_events, endpoint=False))

        lower_limit_event = -max_int_amplitude
        upper_limit_event = max_int_amplitude

        generated_array = np.zeros(len(t))

        for i in range(len(start_index_list)):

            start_index = start_index_list[i]
            if i == len(start_index_list) - 1:
                end_index = len(t) - 1
            else:
                end_index = start_index_list[i+1]

            lower_limit_event = -max_int_amplitude - generated_array[start_index]
            upper_limit_event = max_int_amplitude - generated_array[start_index]

            event_value = (upper_limit_event - lower_limit_event) * env.np_random.random() + lower_limit_event

            increment = event_value / (end_index -  start_index)
            increment = min(max(increment, -max_slope), max_slope) 
            event_profile = ramp_profile(t, start_index, end_index, increment)
            
            generated_array += event_profile

        d_generated_array = np.diff(generated_array) / env.delta_t
            
        return d_generated_array
    return random_d_ramps_generator
"""
Create a function that takes an AeroGym environment and generates a numpy array with a size corresponding to the maximum number of timesteps in the passed environment and containing the derivative of a random trajectory of at most `n_events_max` steps and ramps. The derivative has a maximum amplitude of `max_d_amplitude` and its integral has a maximum amplitude of `max_int_amplitude`. 
"""
def random_d_steps_ramps(n_events_max=20, max_int_amplitude=1.0, max_d_amplitude=1.0):
    def random_d_steps_ramps_generator(env):
    
        t = np.linspace(0, env.t_max, int(env.t_max / env.delta_t) + 1)

        # compute max int event amplitudes to avoid exceeding max_d_amplitude
        max_slope = max_d_amplitude * env.delta_t
        max_step = max_d_amplitude * env.delta_t

        # generate list of at most n_event_max list indices where the events take place
        n_events = env.np_random.integers(0, high=n_events_max, endpoint=True)
        start_index_list = np.sort(env.np_random.integers(0, high=len(t)-2, size=n_events, endpoint=False))

        lower_limit_event = -max_int_amplitude
        upper_limit_event = max_int_amplitude

        generated_array = np.zeros(len(t))

        for i in range(len(start_index_list)):

            start_index = start_index_list[i]
            if i == len(start_index_list) - 1:
                end_index = len(t) - 1
            else:
                end_index = start_index_list[i+1]

            lower_limit_event = -max_int_amplitude - generated_array[start_index]
            upper_limit_event = max_int_amplitude - generated_array[start_index]

            event_value = (upper_limit_event - lower_limit_event) * env.np_random.random() + lower_limit_event

            # generate a random integer to assign an event type
            event_type = env.np_random.integers(2, endpoint=False)

            if event_type == 0:
                event_value = min(max(event_value, -max_step), max_step) 
                event_profile = step_profile(t, start_index, event_value)
            else:
                increment = event_value / (end_index -  start_index)
                increment = min(max(increment, -max_slope), max_slope) 
                event_profile = ramp_profile(t, start_index, end_index, increment)
            
            generated_array += event_profile

        d_generated_array = np.diff(generated_array) / env.delta_t
            
        return d_generated_array
    return random_d_steps_ramps_generator

"""
Create a function that takes an AeroGym environment and generates a numpy array with a size corresponding to the maximum number of timesteps in the passed environment and containing a trajectory consisting of a random number (up to `n_modes_max`) of fourier sine modes where the n-th mode has a frequency of 1/`T` and random amplitudes between 0 and 1/n. 
"""
def random_fourier_series(T=30, n_modes_max=100):
    def random_fourier_series_generator(env):
        t = np.linspace(0, env.t_max, int(env.t_max / env.delta_t) + 1)
        
        n_modes = env.np_random.integers(1, high=n_modes_max)
        A = env.np_random.normal(0, 1, n_modes)

        s = 0.0
        s += sum(np.sin(2*math.pi/T*n*t)*A[n]/(n+1) for n in range(0, n_modes))
        return s
    return random_fourier_series_generator

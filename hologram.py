import numpy as np



# Frequency 1 square wave, variable phase shift in radians, variable duty
# cycle in [0, 1].
def square_wave(t, phase_shift, duty):
    phase_norm = t-phase_shift/(2*np.pi)

    return (np.mod(phase_norm+duty/2, 1) < duty).astype(float)

# Generate a binary parallel Lee hologram using duty cycle modulation.
#     ampl  - Amplitude of the target field, in the range 0 to 1
#     phase - Phase of the target field, in radians
#     freq  - 2-component frequency vector of the carrier fringes
# Returns a binary-valued hologram image of the same shape as ampl & phase
def parallel_lee(ampl, phase, freq):
    if ampl.shape != phase.shape:
        raise ValueError("Ampl and phase arrays must have the same shape")

    ys, xs = map(np.arange, ampl.shape)
    xx, yy = np.meshgrid(xs, ys)
    duty = np.arcsin(ampl)/np.pi
    
    return square_wave(xx*freq[0]+yy*freq[1], phase, duty)

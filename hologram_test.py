import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage



# Frequency 1 square wave, variable phase shift in radians, variable duty
# cycle in [0, 1].
def square_wave(t, phase_shift, duty):
    phase_norm = t-phase/(2*np.pi)

    return (np.mod(phase_norm+duty/2, 1) < duty).astype(float)

def gen_hologram(ampl, phase, freq):
    ys, xs = map(np.arange, ampl.shape)
    xx, yy = np.meshgrid(xs, ys)
    duty = np.arcsin(ampl)/np.pi
    
    return square_wave(xx*freq[0]+yy*freq[1], phase, duty)

def imshow_complex(z):
    hue = 0.5+np.angle(z)/(2*np.pi)
    value = np.abs(z)/np.abs(z).max()
    hsv = np.dstack([hue, np.ones_like(hue), value])
    rgb = plt.cm.colors.hsv_to_rgb(hsv)

    plt.imshow(rgb)

# Size of the hologram and image
size = (800, 1280)
freq = (0.106331, 0.0657164)
cutoff_frac = 0.1

xs = np.arange(size[1])-size[1]//2
ys = np.arange(size[0])-size[0]//2
xx, yy = np.meshgrid(xs, ys)
phase = np.arctan2(yy, xx)
radius = np.hypot(xx, yy)
radius_lim = min(size)/2
ampl = 0.5-0.5*np.cos(2*np.pi*radius/radius_lim)
ampl[radius > radius_lim] = 0

holo = gen_hologram(ampl, phase, freq)

ft = np.fft.fft2(holo)
fy, fx = map(np.fft.fftfreq, size)
fxx, fyy = np.meshgrid(fx, fy)

fy, fx = map(np.fft.fftfreq, size)
fxx, fyy = np.meshgrid(fx, fy)
ft_cut = ndimage.shift(ft, (freq[1]*size[0], freq[0]*size[1]), mode = "grid-wrap")
ft_cut[np.hypot(fxx, fyy) > np.linalg.norm(freq)*cutoff_frac] = 0

img = np.fft.ifft2(ft_cut)

plt.subplot(2, 3, 1)
plt.imshow(phase)
plt.subplot(2, 3, 2)
plt.imshow(ampl)
plt.subplot(2, 3, 3)
plt.imshow(holo)
plt.subplot(2, 3, 4)
plt.imshow(np.fft.fftshift(np.log(1+np.abs(ft)**2)))
plt.subplot(2, 3, 5)
plt.imshow(np.fft.fftshift(np.log(1+np.abs(ft_cut)**2)))
plt.subplot(2, 3, 6)
imshow_complex(img)
plt.show()

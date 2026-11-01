"""
Seismic Data Processing Mini-Project (Python)
=============================================
Workflow: synthetic CMP gather -> noise -> band-pass filter -> velocity
analysis (semblance) -> NMO correction -> stack.

Run:  pip install numpy scipy matplotlib
      python seismic_processing_project.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

rng = np.random.default_rng(42)

# ---------- 1. SYNTHETIC DATA (acquisition) ----------
dt = 0.004                       # sampling interval (s) -> 4 ms
nt = 750                         # 3 s of data
t = np.arange(nt) * dt
offsets = np.arange(0, 1500, 50) # receiver offsets (m)

# Three reflectors: zero-offset time (s) and RMS velocity (m/s)
reflectors = [(0.4, 1800), (1.0, 2300), (1.8, 2900)]

def ricker(f, dt, length=0.128):
    """Ricker wavelet: the standard synthetic seismic source pulse."""
    tt = np.arange(-length/2, length/2, dt)
    a = (np.pi * f * tt) ** 2
    return (1 - 2*a) * np.exp(-a)

wavelet = ricker(25, dt)         # 25 Hz dominant frequency

gather = np.zeros((nt, len(offsets)))
for ix, x in enumerate(offsets):
    for t0, v in reflectors:
        # Hyperbolic travel-time: t(x) = sqrt(t0^2 + x^2 / v^2)
        tx = np.sqrt(t0**2 + (x / v) ** 2)
        gather[int(tx / dt), ix] += 1.0
    gather[:, ix] = np.convolve(gather[:, ix], wavelet, mode="same")

clean = gather.copy()

# ---------- 2. ADD NOISE ----------
# Random noise + low-frequency "ground roll" (strong, slow, low-frequency)
gather += 0.6 * rng.standard_normal(gather.shape)
for ix, x in enumerate(offsets):
    tg = x / 400 + 0.2                    # ground roll arrives at ~400 m/s
    i = int(tg / dt)
    if i < nt - 60:
        gr = ricker(8, dt)[:60] * 3
        gather[i:i+len(gr), ix] += gr
noisy = gather.copy()

# ---------- 3. PRE-PROCESSING: band-pass filter (signal processing) ----------
def bandpass(data, f_low, f_high, dt, order=4):
    nyq = 0.5 / dt
    b, a = butter(order, [f_low/nyq, f_high/nyq], btype="band")
    return filtfilt(b, a, data, axis=0)

filtered = bandpass(noisy, 12, 60, dt)

# Amplitude recovery: boost later times (waves weaken with depth)
gain = (t ** 1.0)[:, None] + 0.2
filtered_gained = filtered * gain

# ---------- 4. VELOCITY ANALYSIS (semblance) ----------
def nmo_correct(gath, offs, t, v, dt, stretch_mute=0.5):
    out = np.zeros_like(gath)
    for ix, x in enumerate(offs):
        t_nmo = np.sqrt(t**2 + (x / v) ** 2)   # time to read from
        out[:, ix] = np.interp(t_nmo, t, gath[:, ix], left=0, right=0)
        # mute samples stretched too much
        stretch = (t_nmo - t) / np.maximum(t, 1e-6)
        out[stretch > stretch_mute, ix] = 0
    return out

def semblance(gath, offs, t, vels, dt, win=10):
    sem = np.zeros((len(t), len(vels)))
    for iv, v in enumerate(vels):
        c = nmo_correct(gath, offs, t, v, dt)
        num = c.sum(axis=1) ** 2
        den = (c ** 2).sum(axis=1) * c.shape[1] + 1e-12
        k = np.ones(win)
        sem[:, iv] = np.convolve(num, k, "same") / np.convolve(den, k, "same")
    return sem

vels = np.arange(1500, 3600, 50)
sem = semblance(filtered_gained, offsets, t, vels, dt)

# Pick the velocity at each reflector's time (automatic max-semblance pick)
picks = []
for t0, v_true in reflectors:
    it = int(t0 / dt)
    sl = sem[max(it-8, 0):it+8, :].max(axis=0)
    picks.append(vels[sl.argmax()])
print("True velocities  :", [v for _, v in reflectors])
print("Picked velocities:", picks)

# Build a velocity function by interpolating the picks in time
t_picks = [r[0] for r in reflectors]
v_func = np.interp(t, t_picks, picks)

# ---------- 5. NMO CORRECTION + STACK ----------
def nmo_correct_vfunc(gath, offs, t, vf, dt):
    out = np.zeros_like(gath)
    for ix, x in enumerate(offs):
        t_nmo = np.sqrt(t**2 + (x / vf) ** 2)
        out[:, ix] = np.interp(t_nmo, t, gath[:, ix], left=0, right=0)
        stretch = (t_nmo - t) / np.maximum(t, 1e-6)
        out[stretch > 0.5, ix] = 0
    return out

nmo = nmo_correct_vfunc(filtered_gained, offsets, t, v_func, dt)
stack = nmo.sum(axis=1) / np.maximum((nmo != 0).sum(axis=1), 1)
single_trace = filtered_gained[:, 0]

# ---------- 6. QC PLOTS ----------
def show(ax, d, title):
    clip = np.percentile(np.abs(d), 98)
    ax.imshow(d, aspect="auto", cmap="gray", vmin=-clip, vmax=clip,
              extent=[offsets[0], offsets[-1], t[-1], 0])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Offset (m)")
    ax.set_ylabel("Time (s)")

fig, axs = plt.subplots(1, 5, figsize=(22, 6))
show(axs[0], noisy, "1. Noisy CMP gather")
show(axs[1], filtered_gained, "2. Band-pass 12-60 Hz + gain")
im = axs[2].imshow(sem, aspect="auto", cmap="jet",
                   extent=[vels[0], vels[-1], t[-1], 0])
axs[2].plot(v_func, t, "w--", lw=1.5)
axs[2].set_title("3. Semblance + picked velocities", fontsize=10)
axs[2].set_xlabel("Velocity (m/s)")
show(axs[3], nmo, "4. After NMO correction")
axs[4].plot(stack, t, "k", lw=0.8)
axs[4].plot(single_trace / np.abs(single_trace).max() * np.abs(stack).max(),
            t, color="red", alpha=0.35, lw=0.6, label="single trace")
axs[4].invert_yaxis()
axs[4].legend(fontsize=8)
axs[4].set_title("5. Stack (black) vs single trace", fontsize=10)
axs[4].set_xlabel("Amplitude")
plt.tight_layout()
plt.savefig("seismic_workflow.png", dpi=130)

# Frequency spectrum before/after filtering (Fourier transform demo)
freqs = np.fft.rfftfreq(nt, dt)
fig2, ax = plt.subplots(figsize=(7, 4))
ax.plot(freqs, np.abs(np.fft.rfft(noisy[:, 10])), label="Before", alpha=.7)
ax.plot(freqs, np.abs(np.fft.rfft(filtered[:, 10])), label="After band-pass")
ax.set_xlim(0, 100); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Amplitude")
ax.set_title("Amplitude spectrum, one trace"); ax.legend()
plt.tight_layout(); plt.savefig("spectrum.png", dpi=130)
print("Saved seismic_workflow.png and spectrum.png")

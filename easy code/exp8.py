import numpy as np
import matplotlib.pyplot as plt

# ① SETUP
bits = np.random.randint(0, 2, 100000)
bpsk = 2*bits - 1.0

L = 4
T = 1
beta_list = [0.1, 0.5, 0.7, 0.9]
SNR = [0, 2, 4, 6, 8, 10]

u = np.zeros(len(bpsk) * L)
u[::L] = bpsk
t = np.arange(-4, 4 + 1/L, 1/L)   # N_sym=8 half-window

# ② SRRC FUNCTION
def srrc(t, beta, T):
    p = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0):
            p[i] = (1/np.sqrt(T)) * (1 + beta*(4/np.pi - 1))
        elif np.isclose(abs(ti), T/(4*beta)):
            p[i] = (beta/np.sqrt(2*T)) * (
                (1 + 2/np.pi)*np.sin(np.pi/(4*beta)) +
                (1 - 2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            num = np.sin(np.pi*ti*(1-beta)/T) + (4*beta*ti/T)*np.cos(np.pi*ti*(1+beta)/T)
            den = (np.pi*ti/T) * (1 - (4*beta*ti/T)**2)
            p[i] = (1/np.sqrt(T)) * num/den
    return p

# ③ EYE DIAGRAM FUNCTION
def plot_eye(sig, L, title):
    plt.figure(figsize=(5, 4))
    for i in range(100):
        seg = sig[i*L : i*L + 2*L]
        if len(seg) == 2*L:
            plt.plot(seg)
    plt.title(title); plt.grid(True)

# ④ DOUBLE LOOP
for beta in beta_list:
    p = srrc(t, beta, T)
    tx = np.convolve(u, p, mode='full')
    for snr_db in SNR:
        noise = np.sqrt(np.mean(tx**2) / 10**(snr_db/10)) * np.random.randn(len(tx))
        y = np.convolve(tx + noise, p, mode='full')
        plot_eye(y, L, f"Eye (β={beta}, SNR={snr_db}dB)")

# ⑤ SHOW
plt.show()

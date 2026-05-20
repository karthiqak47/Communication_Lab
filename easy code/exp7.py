import numpy as np
import matplotlib.pyplot as plt

# ① SETUP
bits = np.random.randint(0, 2, 100000)
bpsk = 2*bits - 1.0
L, T, beta = 4, 1, 0.5
SNR = np.array([0, 2, 4, 6, 8, 10])

u = np.zeros(len(bpsk) * L)
u[::L] = bpsk
t = np.arange(-4, 4 + 1/L, 1/L)

# ② SRRC
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

# ③ TRANSMIT
p = srrc(t, beta, T)
tx = np.convolve(u, p, mode='full')
delay = 2 * ((len(p)-1)//2)      # total delay = 2 × one-filter delay
avgpower = np.mean(tx**2)

# ④ BER LOOP
ber = []
for snr_db in SNR:
    noise = np.sqrt(avgpower / 10**(snr_db/10)) * np.random.randn(len(tx))
    y = np.convolve(tx + noise, p, mode='full')
    bits_hat = (y[delay::L][:len(bits)] > 0).astype(int)
    ber.append(np.mean(bits != bits_hat))

# ⑤ PLOT
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1); plt.plot(t, p, 'g'); plt.title('SRRC Pulse (β=0.5)'); plt.grid(True)
plt.subplot(1,2,2); plt.semilogy(SNR, ber, 'r-o'); plt.title('BER vs SNR')
plt.xlabel('SNR (dB)'); plt.ylabel('BER'); plt.grid(True, which='both')
plt.tight_layout(); plt.show()

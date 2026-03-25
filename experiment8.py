import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------- IMAGE ----------------
img = cv2.imread("cameraman.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256,256))

bits = np.unpackbits(img.flatten())

bpsk = 2*bits.astype(float) - 1

plt.scatter(bpsk[:100], np.zeros(100))
plt.title("BPSK Constellation")
plt.show()

# ---------------- PARAMETERS ----------------
L = 4      
T_sym = 1
N_sym = 8
beta = 0.5
SNR = np.array([0, 2, 4, 6, 8, 10])

# ---------------- UPSAMPLE ----------------
u = np.zeros(len(bpsk)*L)
u[::L] = bpsk

t = np.arange(-N_sym/2, N_sym/2 + 1/L, 1/L)

# ---------------- SRRC ----------------
def srrc(t, beta, T):
    p = np.zeros_like(t)

    for i, ti in enumerate(t):

        if np.isclose(ti, 0):
            p[i] = (1/np.sqrt(T)) * (1 + beta*(4/np.pi - 1))

        elif beta != 0 and np.isclose(abs(ti), T/(4*beta)):
            p[i] = (beta/np.sqrt(2*T)) * (
                (1 + 2/np.pi)*np.sin(np.pi/(4*beta)) +
                (1 - 2/np.pi)*np.cos(np.pi/(4*beta))
            )

        else:
            num = np.sin(np.pi*ti*(1-beta)/T) + \
                  (4*beta*ti/T)*np.cos(np.pi*ti*(1+beta)/T)

            den = (np.pi*ti/T) * (1 - (4*beta*ti/T)**2)

            p[i] = (1/np.sqrt(T)) * (num/den)

    return p

p_n = srrc(t, beta, T_sym)

# ---------------- TRANSMIT ----------------
signal = np.convolve(u, p_n, mode='full')

nfilter = len(p_n)
delay = (nfilter - 1)//2
totaldelay = 2*delay

avgpower = np.mean(signal**2)

# ---------------- BER + IMAGE ----------------
berlist = []
reconstructed_images = []

for snr in SNR:
    noise = np.sqrt(avgpower/(10**(snr/10))) * np.random.randn(len(signal))
    r = signal + noise

    y = np.convolve(r, p_n, mode='full')

    yn = y[totaldelay::L]
    yfinal = yn[:len(bpsk)]

    bitsfinal = (yfinal > 0).astype(int)

    ber = np.mean(bits != bitsfinal)
    berlist.append(ber)

    rec_img = np.packbits(bitsfinal).reshape(256,256)
    reconstructed_images.append(rec_img)

# ---------------- EYE DIAGRAM ----------------
def plot_eye(signal, L, nSamples, nTraces, title):
    plt.figure(figsize=(5,4))

    for i in range(nTraces):
        start = i*L
        segment = signal[start:start+nSamples]

        if len(segment) == nSamples:
            plt.plot(segment)

    plt.title(title)
    plt.grid(True)

nSamples = 2*L
nTraces = 100

snr_eye = [0, 4, 8, 10]

for snr_db in snr_eye:
    noise = np.sqrt(avgpower/(10**(snr_db/10))) * np.random.randn(len(signal))
    r_n = signal + noise
    y_n = np.convolve(r_n, p_n, mode='full')

    plot_eye(y_n, L, nSamples, nTraces,
             f"Eye Diagram (β={beta}, SNR={snr_db} dB)")

# ---------------- BETA VARIATION ----------------
beta_list = [0.1, 0.5, 0.7, 0.9]
snr_fixed = 4

for b in beta_list:
    p_beta = srrc(t, b, T_sym)
    s_beta = np.convolve(u, p_beta, 'full')

    noise = np.sqrt(np.mean(s_beta**2)/(10**(snr_fixed/10))) \
            * np.random.randn(len(s_beta))

    r_beta = s_beta + noise
    y_beta = np.convolve(r_beta, p_beta, 'full')

    plot_eye(y_beta, L, nSamples, nTraces,
             f"Eye Diagram (β={b}, SNR=4 dB)")

# ---------------- RESULTS ----------------
plt.figure(figsize=(16,12))

# SRRC
plt.subplot(3,4,1)
plt.plot(t, p_n)
plt.title("SRRC Pulse Shape")
plt.grid(True)

# BER
plt.subplot(3,4,2)
plt.semilogy(SNR, berlist, 'o-')
plt.title("BER vs SNR")
plt.grid(True)

# Original
plt.subplot(3,4,3)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Reconstructed images
for i in range(len(SNR)):
    plt.subplot(3,4,i+4)
    plt.imshow(reconstructed_images[i], cmap='gray')
    plt.title(f"SNR={SNR[i]} dB")
    plt.axis('off')

plt.tight_layout()
plt.show()

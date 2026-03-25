import numpy as np
import matplotlib.pyplot as plt
import cv2

#For task 5:
from scipy.special import erfc

import numpy as np
import cv2

# Read and normalize image
img = cv2.imread("cameraman.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256,256))
M,N=img.shape
x = img / 255.0
x_vec = x.flatten()

def pcm_encode(x, b):
    
    L = 2**b
    delta = 1 / L

    # Quantization index
    k = np.floor(x / delta)
    k = np.clip(k, 0, L-1).astype(np.uint8)

    # Quantized signal
    xcap = (k + 0.5) * delta

    # SQNR
    Ps = np.mean(x**2)
    Pn = np.mean((x - xcap)**2)
    SQNR_dB = 10 * np.log10(Ps / Pn)

    # Convert to bitstream (vectorized)
    b_tx = np.unpackbits(k)

    return k, xcap, b_tx, SQNR_dB

# Run
b = 8
k, xcap, b_tx, SQNR = pcm_encode(x_vec, b)

print("Bitstream length:", len(b_tx))
print("SQNR (dB):", SQNR)


Eb = 1
bipolar = 1 - 2*b_tx.astype(int)  # 0 maps to +1, 1 maps to -1
s = bipolar * np.sqrt(Eb)  #symbols of BPSK Modulated signals

plt.figure()
plt.scatter(s,np.zeros_like(s),s=10)
plt.xlabel("In-Phase")
plt.ylabel("Quadrature-Phase")
plt.title("BPSK Constellation with no Noise Addition")
plt.grid()
plt.show()


list_y = []
Eb = 1
EbN0_dB = np.arange(0, 11, 2)
EbN0_lin = 10**(EbN0_dB / 10)

for EbN0 in EbN0_lin:
    
    noise_var = Eb / (2 * EbN0)
    
    noise = np.sqrt(noise_var) * (
        np.random.randn(len(s)) + 1j*np.random.randn(len(s))
    )
    
    y = s + noise
    list_y.append(y)




# ---------------- RECEIVER ----------------
list_b_rx = [(np.real(y) < 0).astype(np.uint8) for y in list_y]

# ---------------- BER ----------------
sim_BER = [np.mean(b_tx != b_rx) for b_rx in list_b_rx]

# ---------------- PCM DECODE ----------------
def pcm_decode(b_rx, b, M, N):

    total_bits = (len(b_rx)//b)*b
    bits = b_rx[:total_bits].reshape(-1, b)

    # Convert bits → decimal (vectorized)
    k_hat = bits.dot(2**np.arange(b-1, -1, -1))

    # Reconstruction
    delta = 1/(2**b)
    xcap = (k_hat + 0.5) * delta

    return xcap.reshape(M, N)

# ---------------- RECONSTRUCT ----------------
reconstructed_images = [
    pcm_decode(b_rx, b, M, N) for b_rx in list_b_rx
]

# ---------------- PRINT BER ----------------
for EbN0, ber in zip(EbN0_dB, sim_BER):
    print(f"Eb/N0 = {EbN0} dB : BER = {ber:.6f}")

# ---------------- IMAGE PLOTS ----------------
for EbN0, img_hat in zip(EbN0_dB, reconstructed_images):
    plt.figure()
    plt.imshow(img_hat, cmap='gray')
    plt.title(f"Reconstructed Image (Eb/N0 = {EbN0} dB)")
    plt.axis('off')
    plt.show()

# ---------------- I-Q SCATTER ----------------
for EbN0, y in zip(EbN0_dB, list_y):
    plt.figure()
    plt.scatter(np.real(y[:5000]), np.imag(y[:5000]), s=5)
    plt.axvline(0)  # decision boundary
    plt.title(f"Eb/N0 = {EbN0} dB")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid()
    plt.show()

# QPSK using SAME LOGIC as your BPSK code

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

data = np.random.normal(0,1,100000).astype(np.uint8)

# Convert to bitstream
bits = np.unpackbits(data)
print("Bits:")
print(bits)

if len(bits) % 2 != 0:
    bits = np.append(bits, 0)

# Split into I and Q bits
I = 1 - 2*bits[0::2].astype(int)
Q = 1 - 2*bits[1::2].astype(int)

# Complex QPSK symbols
qpsk = I + 1j*Q

print("\nQPSK Symbols:")
print(qpsk)


# CONSTELLATION WITHOUT NOISE

plt.figure()
plt.scatter(np.real(qpsk), np.imag(qpsk))
plt.grid(True)
plt.title("QPSK Constellation")
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")
plt.show()

# -------------------------------------------------
# CHANNEL PARAMETERS
# -------------------------------------------------
snr = [0,2,4,6,8,10,12]

out = []
nber = []

for i in snr:

    EbN0_lin = 10**(i/10)
    sigma = np.sqrt(1/(2*EbN0_lin))
    noise = sigma * (np.random.randn(len(qpsk))+ 1j*np.random.randn(len(qpsk)))
    
    y = qpsk + noise

    out.append(y)

    #demodulation
    rx_I = (np.real(y) < 0).astype(np.uint8)
    rx_Q = (np.imag(y) < 0).astype(np.uint8)

    # Reconstruct bitstream
    brx = np.empty(len(bits), dtype=np.uint8)
    brx[0::2] = rx_I
    brx[1::2] = rx_Q

    # BER
    ber = np.mean(brx != bits)

    nber.append(ber)

    
    plt.figure()
    plt.scatter(np.real(y), np.imag(y), label=f"Eb/N0 = {i}")
    plt.grid(True)
    plt.title(f"QPSK at Eb/N0 = {i} dB")
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.show()


# THEORETICAL BER

EbN0_lin = 10**(np.array(snr)/10)
theoretical_BER = 0.5 * erfc(np.sqrt(EbN0_lin))


# BER PLOT

plt.figure()
plt.semilogy(snr, nber, 'o-', label='Simulated BER')

plt.semilogy(snr,theoretical_BER,"*-",label='Theoretical')

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("BER")

plt.title("QPSK over AWGN")

plt.grid(True)
plt.legend()
plt.show()

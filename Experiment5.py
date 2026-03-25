import numpy as np 
from PIL import Image
from scipy.special import erfc
import matplotlib.pyplot as plt
img  = Image.open(r"/content/cameraman.png").convert("L")
img = img.resize((256,256))

img_np  = np.array(img)
img_norm= img_np/255

b=8
L = 2**b
img_q = np.floor(img_norm *(L-1)).astype(np.uint8)
bitstream = np.unpackbits(img_q.flatten())
reshaped_bits = bitstream.reshape(-1,2)

def qpsk_mapper(bits):
    mapping = {
        (0,0): 1+1j,
        (0,1): -1+1j,
        (1,0): 1-1j,
        (1,1): -1-1j
    }
    return np.array([mapping[tuple(b)] for b in bits])
s = qpsk_mapper(reshaped_bits)
print(s)

plt.scatter(np.real(s[:1000]), np.imag(s[:1000]), s=5)
plt.axhline(0); plt.axvline(0)
plt.show()

Eb = 1
EbNdb = np.array([0,2,4,6,8,10])
EbNlin = 10**(EbNdb/10)

def awgn_channel(s, EbNlin):
    No = Eb/EbNlin
    sigma2 = No/2
    noise = np.sqrt(sigma2) * (np.random.randn(len(s)) + 1j*np.random.randn(len(s)))
    return s+noise,sigma2

def qpsk_demapper(y):
    bits = np.zeros((len(y), 2), dtype=np.uint8)

    real = np.real(y)
    imag = np.imag(y)

    # Quadrant-based Gray decoding
    bits[(real >= 0) & (imag >= 0)] = [0, 0]
    bits[(real < 0)  & (imag >= 0)] = [0, 1]
    bits[(real < 0)  & (imag < 0)]  = [1, 1]
    bits[(real >= 0) & (imag < 0)]  = [1, 0]

    return bits

def compute_ber(tx_bits, rx_bits):
    errors = np.sum(tx_bits != rx_bits)
    return errors / len(tx_bits)

for i in range(len(EbNlin)):
    y, sigma2 = awgn_channel(s, EbNlin[i])
    rx_bits_pairs = qpsk_demapper(y)
    rx_bits = rx_bits_pairs.flatten()
    ff= str(i)
    
    ber_0db = compute_ber(bitstream, rx_bits[:len(bitstream)])
    print("BER at "+str(EbNdb[i])+"dB:", ber_0db)
    plt.scatter(np.real(y[:3000]), np.imag(y[:3000]))
    plt.axhline(0); plt.axvline(0)
    plt.title("db constallation:"+str(EbNdb[i])+"db")
    
    plt.show()


ber_sim = []

for snr in EbNlin :
    y,_ = awgn_channel(s,snr)
    rx_bits = qpsk_demapper(y).flatten()
    ber_sim.append(compute_ber(bitstream,rx_bits[:len(bitstream)]))
ber_sim = np.array(ber_sim)


ber_theory = 0.5 * erfc(np.sqrt(EbNlin))
plt.figure(figsize=(7,5))

plt.semilogy(EbNdb, ber_sim, 'o', label='Sim BER')
plt.semilogy(EbNdb, ber_theory, '', label='Theory BER')

plt.xlabel('EbNdb')
plt.ylabel('BER')
plt.title('BER vs Eb/N0 for QPSK over AWGN')

plt.grid(True, which='both')
plt.legend()
plt.show()

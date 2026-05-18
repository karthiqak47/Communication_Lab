#exp 3
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
N=100000
snr=[2,4,6,8,10,12]
data = np.random.randint(0, 256, 100000, dtype=np.uint8)
bits = np.unpackbits(data)
print(bits)

#bpsq
bpsk=1-2*bits.astype(int)
print(bpsk)

plt.scatter(bpsk,np.zeros(len(bpsk)))
plt.grid(True)
plt.show()
out=[]
nber=[]
for i in snr:
  ebnlin= 10**(i/10)
  sigma=1/(2*ebnlin)
  noise = np.sqrt(sigma) * ( np.random.randn(len(bpsk)) +1j*np.random.randn(len(bpsk)))

  y=bpsk+noise
  out.append(y)
  brx=(np.real(y)<0).astype(np.uint8)
  ber=np.mean(brx!=bits)
  nber.append(ber)

EbN0_lin = 10**(np.array(snr)/ 10) 
theoretical_BER = 0.5 * erfc(np.sqrt(EbN0_lin))
plt.semilogy(snr, nber, 'o-', label='Simulated BER')
plt.semilogy(snr, theoretical_BER, 's-', label='Theoretical BER')

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("BER")

plt.title("BPSK over AWGN")

plt.grid(True, which='both')
plt.legend()
plt.show()


for i,y in zip(snr,out):
  plt.scatter(y.real,y.imag,label=i,s=1)
  plt.grid(True)
  plt.show()









import numpy as np
import matplotlib.pyplot as plt


# Define the number of bits directly
N_bits = 100000
bits = np.random.randint(0, 2, N_bits) # Generate random bits (0s and 1s) directly

bpsk=2*bits.astype(float)-1



L = 4
T_sym = 1
N_sym = 8
beta = 0.5
SNR = np.array([0, 2, 4, 6, 8, 10])


#upsample
u =np.zeros(len(bpsk)*L)
u[::L]=bpsk

t=np.arange(-N_sym/2,N_sym/2+1/L,1/L)

def srrc(t, beta, T):
    p = np.zeros_like(t)

    for i, ti in enumerate(t):

        if np.isclose(ti, 0):
            p[i] = (1/np.sqrt(T)) * (1 + beta*(4/np.pi - 1))

       
        elif np.isclose(abs(ti), T/(4*beta)):
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
signal=np.convolve(u,p_n,mode='full')
nfilter=len(p_n)
delay=(nfilter-1)//2
totaldelayy=2*delay
avgpower=np.mean(signal**2)


berlist=[]

for snr_val in SNR : # Renamed 'snr' to 'snr_val' to avoid conflict with SNR array
  noise = np.sqrt(avgpower / (10**(snr_val/10))) * np.random.randn(len(signal))
  r = signal + noise

  y=np.convolve(r,p_n,mode='full')
  yn=y[totaldelayy::L]
  yfinal=yn[:len(bpsk)]

  bitsfinal= (yfinal>0).astype(int)
  ber=np.mean(bits!=bitsfinal)
  berlist.append(ber)


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.plot(t, p_n, 'g')
plt.title(r'SRRC Pulse Shape ($\beta=0.5$)') # Fixed typo in beta
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(SNR, berlist, 'r-o')
plt.title("BER vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid(True, which='both')

plt.tight_layout()
plt.show()

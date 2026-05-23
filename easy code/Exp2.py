import numpy as np
import matplotlib.pyplot as plt
fs=8000
t=np.arange(0,1/200,1/fs)
signal=5*np.sin(2*np.pi*200*t)
plt.plot(t,signal)
plt.show()

def pcm(l,x,a):
  b=int(np.log2(l))
  delta=(2*a)/l
  k=np.floor((x+a)/delta)
  k=np.clip(k,0,l-1)
  xhat=-a+(k+0.5)*delta
  pq=np.mean((x-xhat)**2)
  ps=np.mean(x**2)
  snr_sim=10*np.log10(ps/pq)
  snr_theo=6.02*b+1.76
  return xhat,pq,snr_sim,snr_theo,b


l=[2,4,8,16,32,64]
out=[]
snrsim=[]
snrtheo=[]
for i in l:
  xhat,pq,snr_sim,snr_theo,b=pcm(i,signal,5)
  out.append(xhat)
  snrsim.append(snr_sim)
  snrtheo.append(snr_theo)
b=[1,2,3,4,5,6]
for i in l:
  plt.plot(b,snrsim,"s-")
  plt.plot(b,snrtheo)
plt.show()

plt.plot(t[:20],signal[:20])
plt.step(t[:20],xhat[:20])
plt.show()

import numpy as np
import matplotlib.pyplot as plt

r=41
N=100000
X=np.random.normal(r,1,N)
Y=np.random.normal(r,1,N)
plt.hist(X,bins=100)
plt.show()
plt.hist(Y,bins=100,color="red")
plt.show()

Z=X+Y
plt.hist(Z,bins=100)
plt.show()

hx,binx=np.histogram(X,bins=100,density=True)
hy=np.histogram(Y,bins=100,density=True)
plt.plot(binx[:-1],hx)
plt.show()



#cdf 
cdfx=np.cumsum(hx)
normcdfx=cdfx * (binx[1]-binx[0])
plt.plot(binx[:-1],normcdfx)
plt.show()

rule1=np.sum(np.abs(X-r)<1)
p=rule1/N
print(p)

plt.scatter(X,Y)
plt.show()

#correlation of x and y
x=np.correlate(X,Y,mode="full")
plt.plot(x)
plt.show()

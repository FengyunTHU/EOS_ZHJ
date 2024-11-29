from math import log

y1 = 3.0/100
y2 = 2.7/100
y1p = 3.0/100
y2p = 1.2/100
x = (20/103)*(y1-y2)
zi = log(y1p)-log(y2p)
m = 2560/101.325
print(m)
mu = log(y1-m*x)-log(y2-m*x)
print(zi)
print(mu)
print(zi/mu)
print(x)
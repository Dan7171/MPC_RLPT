# from math import exp

import numpy as np
import matplotlib.pyplot as plt

def expo(x,a,b):
	return np.exp(a*(-x+b))
x = np.linspace(0,1,200) # represents position error (l2 distance) between ee to ee goal 
a_b_pairs = [(1,0.05), (1,0.1), (20,0.05), (20,0.1),(50,0.05)]

for p in a_b_pairs:
	a,b = p
	y = expo(x,a,b) # y = position reward(x)
	plt.plot(x,y, label=f'e^({a}(-x+{b}))')
 
plt.xlabel('x (ee position error)')
plt.ylabel('e^(-a(x+b)) (position reward of x)') 
plt.legend()
plt.show()




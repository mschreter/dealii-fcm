import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("quadrature_points.csv")

fig, ax = plt.subplots(1,1)
ax.scatter(data[:,0], data[:,1], s=0.1)
ax.set_aspect('equal', 'box')
fig.show()
fig.savefig("quadrature_points.pdf")

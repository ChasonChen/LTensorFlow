import matplotlib.pyplot as plt
import numpy as np

# x = np.linspace(-1, 1, 40)
# y = x * x + 5
# plt.plot(x, y)
# plt.gca().spines['right'].set_color('none')
# plt.gca().spines['top'].set_color('none')
# plt.gca().xaxis.set_ticks_position('bottom')
# plt.gca().yaxis.set_ticks_position('left')
# plt.gca().spines['bottom'].set_position(('data', 5))
# plt.gca().spines['left'].set_position(('data', 0))
# plt.show()


n = 1024
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)
t = np.arctan2(x, y)
plt.scatter(x, y, s=75, c=t, alpha=0.5)
# plt.scatter(np.arange(5),np.arange(5))
# plt.xlim((-1, 1))
# plt.ylim((-1, 1))
plt.show()

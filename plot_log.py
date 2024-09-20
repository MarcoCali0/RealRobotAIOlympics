import matplotlib.pyplot as plt
import numpy as np

scores_train = np.loadtxt('scores_train')
scores_test = np.loadtxt('scores_test')

plt.figure()

plt.plot(np.arange(len(scores_train)), scores_train, label='train')
plt.plot(np.arange(len(scores_test)), scores_test, label='test')
plt.grid()
plt.legend()
plt.savefig('log_SNES.pdf')
plt.show()

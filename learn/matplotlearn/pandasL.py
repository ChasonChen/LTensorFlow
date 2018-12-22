import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(np.arange(100, 112).reshape(4, 3), columns=['a', 'b', 'c'])
print df

df.plot()
plt.show()

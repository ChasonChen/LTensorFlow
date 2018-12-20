import numpy as np

array = np.empty((3, 4))
add_arr = [1, 3, 3]
array[0, :add_arr.__len__()] = add_arr
print (array)

import numpy as np

array1 = np.array([[1, 2],[3, 4]],
                  [[1, 2],[3, 4]])

array2 = np.array([[5, 6]])

result = np.concatenate(array1, 0)
print(result)

import numpy as np

array1 = np.array([[1, 2],
                   [3, 4]])
array2 = np.array([[5,6],[0,0]])

result = np.concatenate((array1, array2), axis=1)
print(result)


# 创建一个二维数组
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# 沿着0轴计算每列的和
sum_along_axis_0 = np.sum(arr, axis=0)
print("Sum along axis 0:", sum_along_axis_0)

# 沿着1轴计算每行的和
sum_along_axis_1 = np.sum(arr, axis=1)
print("Sum along axis 1:", sum_along_axis_1)
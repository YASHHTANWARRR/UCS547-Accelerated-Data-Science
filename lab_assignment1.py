import numpy as np

print("----- Question 1 -----")
arr = np.array([1, 2, 3, 6, 4, 5])
reversed_arr = arr[::-1]
print(arr)
print(reversed_arr)

print("\n----- Question 2 -----")
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[1, 2], [3, 4]])
print(arr1 == arr2)
print(np.array_equal(arr1, arr2))

print("\n----- Question 3(i) -----")
x = np.array([1,2,3,4,5,1,2,1,1,1])
values_x, counts_x = np.unique(x, return_counts=True)
most_frequent_x = values_x[np.argmax(counts_x)]
indices_x = np.where(x == most_frequent_x)
print(most_frequent_x)
print(indices_x)

print("\n----- Question 3(ii) -----")
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])
values_y, counts_y = np.unique(y, return_counts=True)
most_frequent_y = values_y[np.argmax(counts_y)]
indices_y = np.where(y == most_frequent_y)
print(most_frequent_y)
print(indices_y)

print("\n----- Question 4 -----")
gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]')
print(np.sum(gfg))
print(np.sum(gfg, axis=1))
print(np.sum(gfg, axis=0))
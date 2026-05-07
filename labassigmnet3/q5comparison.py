import matplotlib.pyplot as plt

methods = [
    "CPU",
    "CUDA",
    "Thrust",
    "RAPIDS"
]

times = [
    120,
    15,
    10,
    8
]

plt.figure(figsize=(8,5))

plt.bar(methods, times)

plt.xlabel("Method")
plt.ylabel("Execution Time (ms)")
plt.title("Performance Comparison")

plt.show()
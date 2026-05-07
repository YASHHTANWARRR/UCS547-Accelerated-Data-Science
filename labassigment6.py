import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

import xgboost as xgb

import cudf
import cupy as cp

from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.model_selection import train_test_split as cu_train_test_split
from cuml.metrics import accuracy_score as cu_accuracy

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns = [
    "age","workclass","fnlwgt","education","education_num",
    "marital_status","occupation","relationship","race",
    "sex","capital_gain","capital_loss","hours_per_week",
    "native_country","income"
]

df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)

continuous_features = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week"
]

categorical_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "native_country"
]

binary_features = ["sex"]

target = "income"

for col in continuous_features:
    print(f"{col} --> Continuous (Eligible for GPU Histogram Binning)")

for col in categorical_features:
    print(f"{col} --> Categorical")

for col in binary_features:
    print(f"{col} --> Binary")

print(f"{target} --> Binary Target")

df = df.dropna()

for col in categorical_features + binary_features + [target]:
    df[col] = df[col].astype("category").cat.codes

X = df.drop(target, axis=1)
y = df[target]

gdf = cudf.DataFrame.from_pandas(df)

feature_name = "age"

cpu_hist, cpu_bins = np.histogram(
    df[feature_name],
    bins=20
)

gpu_array = cp.asarray(gdf[feature_name].values)

gpu_hist, gpu_bins = cp.histogram(
    gpu_array,
    bins=20
)

gpu_hist_cpu = cp.asnumpy(gpu_hist)

hist_table = pd.DataFrame({
    "CPU Histogram": cpu_hist,
    "GPU Histogram": gpu_hist_cpu
})

print(hist_table)

plt.figure(figsize=(10,5))

plt.plot(cpu_hist, label="CPU Histogram")
plt.plot(gpu_hist_cpu, label="GPU Histogram")

plt.title("CPU vs GPU Histogram")
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.legend()
plt.show()

quantiles = cp.quantile(
    gpu_array,
    q=cp.linspace(0,1,6)
)

quantiles_cpu = cp.asnumpy(quantiles)

print(quantiles_cpu)

binned = cp.digitize(gpu_array, quantiles)

binned_cpu = cp.asnumpy(binned)

print(pd.Series(binned_cpu).value_counts().sort_index())

plt.figure(figsize=(8,5))
plt.hist(binned_cpu, bins=5)
plt.title("Quantile Binning Distribution")
plt.xlabel("Bin")
plt.ylabel("Count")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

start = time.time()

cpu_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

cpu_rf.fit(X_train, y_train)

cpu_train_time = time.time() - start

start = time.time()

cpu_preds = cpu_rf.predict(X_test)

cpu_pred_time = time.time() - start

cpu_acc = accuracy_score(y_test, cpu_preds)

print(cpu_acc)
print(cpu_train_time)
print(cpu_pred_time)

gX = cudf.DataFrame.from_pandas(X)
gy = cudf.Series(y)

gX_train, gX_test, gy_train, gy_test = cu_train_test_split(
    gX,
    gy,
    test_size=0.2,
    random_state=42
)

start = time.time()

gpu_rf = cuRF(
    n_estimators=100,
    max_depth=16
)

gpu_rf.fit(gX_train, gy_train)

gpu_train_time = time.time() - start

start = time.time()

gpu_preds = gpu_rf.predict(gX_test)

gpu_pred_time = time.time() - start

gpu_acc = cu_accuracy(gy_test, gpu_preds)

print(float(gpu_acc))
print(gpu_train_time)
print(gpu_pred_time)

speedup = cpu_train_time / gpu_train_time

print(speedup)

xgb_model = xgb.XGBClassifier(
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    n_estimators=100,
    max_depth=6
)

start = time.time()

xgb_model.fit(X_train, y_train)

xgb_train_time = time.time() - start

start = time.time()

xgb_preds = xgb_model.predict(X_test)

xgb_pred_time = time.time() - start

xgb_acc = accuracy_score(y_test, xgb_preds)

print(xgb_acc)
print(xgb_train_time)
print(xgb_pred_time)

comparison_table = pd.DataFrame({
    "Model": ["CPU RandomForest", "cuML RandomForest", "XGBoost gpu_hist"],
    "Train Time": [cpu_train_time, gpu_train_time, xgb_train_time],
    "Prediction Time": [cpu_pred_time, gpu_pred_time, xgb_pred_time],
    "Accuracy": [cpu_acc, float(gpu_acc), xgb_acc]
})

print(comparison_table)

plt.figure(figsize=(8,5))

plt.bar(
    comparison_table["Model"],
    comparison_table["Train Time"]
)

plt.title("Training Time Comparison")
plt.ylabel("Seconds")
plt.show()

data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

gX_train = cudf.DataFrame.from_pandas(
    pd.DataFrame(X_train)
)

gX_test = cudf.DataFrame.from_pandas(
    pd.DataFrame(X_test)
)

gy_train = cudf.Series(y_train)
gy_test = cudf.Series(y_test)

start = time.time()

cpu_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

cpu_model.fit(X_train, y_train)

cpu_train_time = time.time() - start

start = time.time()

cpu_preds = cpu_model.predict(X_test)

cpu_pred_time = time.time() - start

cpu_acc = accuracy_score(y_test, cpu_preds)

print(cpu_acc)
print(cpu_train_time)
print(cpu_pred_time)

start = time.time()

gpu_model = cuRF(
    n_estimators=100,
    max_depth=16
)

gpu_model.fit(gX_train, gy_train)

gpu_train_time = time.time() - start

start = time.time()

gpu_preds = gpu_model.predict(gX_test)

gpu_pred_time = time.time() - start

gpu_acc = cu_accuracy(gy_test, gpu_preds)

print(float(gpu_acc))
print(gpu_train_time)
print(gpu_pred_time)

gpu_speedup = cpu_train_time / gpu_train_time

print(gpu_speedup)

tree_counts = [1, 10, 50, 100]

cpu_times = []
gpu_times = []

for trees in tree_counts:

    start = time.time()

    model_cpu = RandomForestClassifier(
        n_estimators=trees
    )

    model_cpu.fit(X_train, y_train)

    cpu_times.append(
        time.time() - start
    )

    start = time.time()

    model_gpu = cuRF(
        n_estimators=trees
    )

    model_gpu.fit(gX_train, gy_train)

    gpu_times.append(
        time.time() - start
    )

plt.figure(figsize=(8,5))

plt.plot(
    tree_counts,
    cpu_times,
    marker='o',
    label='CPU'
)

plt.plot(
    tree_counts,
    gpu_times,
    marker='o',
    label='GPU'
)

plt.xlabel("Number of Trees")
plt.ylabel("Training Time")
plt.title("Forest-Level Parallelism")
plt.legend()
plt.show()

importances = cpu_model.feature_importances_

feature_table = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(
    by="Importance",
    ascending=False
)

print(feature_table.head(10))

plt.figure(figsize=(10,6))

plt.barh(
    feature_table["Feature"][:10],
    feature_table["Importance"][:10]
)

plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.show()

sizes = [100, 200, 400, len(X_train)]

gpu_size_times = []

for size in sizes:

    subset_X = gX_train.iloc[:size]
    subset_y = gy_train.iloc[:size]

    start = time.time()

    model = cuRF(n_estimators=100)

    model.fit(subset_X, subset_y)

    elapsed = time.time() - start

    gpu_size_times.append(elapsed)

plt.figure(figsize=(8,5))

plt.plot(
    sizes,
    gpu_size_times,
    marker='o'
)

plt.xlabel("Dataset Size")
plt.ylabel("Training Time")
plt.title("Data-Level Parallelism")
plt.show()

gpu_feature = cp.asarray(X_train[:,0])

gpu_hist, bins = cp.histogram(
    gpu_feature,
    bins=20
)

gpu_hist_cpu = cp.asnumpy(gpu_hist)

plt.figure(figsize=(8,5))

plt.bar(
    range(len(gpu_hist_cpu)),
    gpu_hist_cpu
)

plt.title("GPU Histogram Binning")
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.show()

final_table = pd.DataFrame({
    "Implementation": ["CPU RandomForest", "GPU cuML RandomForest"],
    "Train Time": [cpu_train_time, gpu_train_time],
    "Prediction Time": [cpu_pred_time, gpu_pred_time],
    "Accuracy": [cpu_acc, float(gpu_acc)]
})

print(final_table)

plt.figure(figsize=(8,5))

plt.bar(
    final_table["Implementation"],
    final_table["Train Time"]
)

plt.title("CPU vs GPU Training Time")
plt.ylabel("Seconds")
plt.show()

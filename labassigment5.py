import os
import cv2
import time
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cupy as cp
import torch
import torchvision.transforms.functional as TF

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types

IMAGE_DIR = "./images"
image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))

assert len(image_paths) >= 20, "Please place at least 20 JPEG images in ./images"

print(f"\nTotal Images Found: {len(image_paths)}")


print("PART 1 : CPU vs GPU IMAGE PROCESSING PIPELINE")
def cpu_pipeline(image_paths):

    processed = []

    start = time.time()

    for path in image_paths:

        img = cv2.imread(path)

        img = cv2.resize(img, (512, 512))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        processed.append(gray)

    total_time = time.time() - start

    return processed, total_time


def gpu_pipeline(image_paths):

    processed = []

    start = time.time()

    for path in image_paths:

        img = cv2.imread(path)

        gpu_img = cp.asarray(img)

        gpu_img = cp.array(
            cv2.resize(cp.asnumpy(gpu_img), (512, 512))
        )

        gray = (
            0.299 * gpu_img[:, :, 2] +
            0.587 * gpu_img[:, :, 1] +
            0.114 * gpu_img[:, :, 0]
        )

        processed.append(gray)

    cp.cuda.Stream.null.synchronize()

    total_time = time.time() - start

    return processed, total_time


cpu_output, cpu_total = cpu_pipeline(image_paths)
gpu_output, gpu_total = gpu_pipeline(image_paths)

cpu_avg = cpu_total / len(image_paths)
gpu_avg = gpu_total / len(image_paths)

speedup = cpu_total / gpu_total

results1 = pd.DataFrame({
    "Pipeline": ["CPU Pipeline", "GPU Pipeline"],
    "Total Time (s)": [cpu_total, gpu_total],
    "Avg Time/Image (s)": [cpu_avg, gpu_avg]
})

print("\nPIPELINE RESULTS")
print(results1)

print(f"\nSpeedup = {speedup:.2f}x")

plt.figure(figsize=(8,5))
plt.bar(results1["Pipeline"], results1["Total Time (s)"])
plt.ylabel("Execution Time (Seconds)")
plt.title("CPU vs GPU Pipeline Execution Time")
plt.show()

print("INSIGHT QUESTION")


print("PART 2 : GRAYSCALE EXPERIMENT")


img1 = image_paths[0]
img2 = image_paths[1]

images_test = [img1, img2]

for idx, path in enumerate(images_test):

    print(f"\nProcessing Image {idx+1}")

    img = cv2.imread(path)

    h, w = img.shape[:2]

    print(f"Resolution: {w} x {h}")

    gray_direct = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]

    gray_manual = (
        0.299 * R +
        0.587 * G +
        0.114 * B
    ).astype(np.uint8)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(gray_direct, cmap='gray')
    plt.title("Direct Grayscale")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(gray_manual, cmap='gray')
    plt.title("Manual Grayscale")
    plt.axis("off")

    plt.show()

print("JPEG THEORY")



print("PART 3 : HYBRID vs DALI PIPELINE")

BATCH_SIZES = [1, 4, 8]
RESOLUTIONS = [(256,256), (512,512)]

def hybrid_pipeline(image_paths, resize_shape):

    start = time.time()

    for path in image_paths:

        img = cv2.imread(path)

        tensor = torch.from_numpy(img).cuda().float()

        tensor = tensor.permute(2,0,1).unsqueeze(0)

        tensor = torch.nn.functional.interpolate(
            tensor,
            size=resize_shape,
            mode='bilinear'
        )

        tensor = tensor / 255.0

    torch.cuda.synchronize()

    total_time = time.time() - start

    return total_time


class DALIPipeline(Pipeline):

    def __init__(self, batch_size, num_threads, device_id, files, resize_shape):
        super().__init__(batch_size,
                         num_threads,
                         device_id,
                         seed=12)

        self.input = fn.readers.file(files=files)

        self.resize_shape = resize_shape

    def define_graph(self):

        jpegs, labels = self.input(name="Reader")

        images = fn.decoders.image(
            jpegs,
            device="mixed",
            output_type=types.RGB
        )

        images = fn.resize(
            images,
            resize_x=self.resize_shape[0],
            resize_y=self.resize_shape[1]
        )

        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT
        )

        return images


records = []

for batch in BATCH_SIZES:

    for res in RESOLUTIONS:

        print(f"\nBatch={batch}, Resolution={res}")

        hybrid_time = hybrid_pipeline(image_paths, res)

        hybrid_throughput = len(image_paths) / hybrid_time

        pipe = DALIPipeline(
            batch_size=batch,
            num_threads=2,
            device_id=0,
            files=image_paths,
            resize_shape=res
        )

        pipe.build()

        start = time.time()

        for _ in range(math.ceil(len(image_paths)/batch)):
            pipe.run()

        dali_time = time.time() - start

        dali_throughput = len(image_paths) / dali_time

        records.append([
            batch,
            str(res),
            hybrid_time,
            dali_time,
            hybrid_throughput,
            dali_throughput
        ])

results2 = pd.DataFrame(records, columns=[
    "Batch Size",
    "Resolution",
    "Hybrid Time",
    "DALI Time",
    "Hybrid Throughput",
    "DALI Throughput"
])

print("\nDALI vs Hybrid Results")
print(results2)

plt.figure(figsize=(10,6))

x = np.arange(len(results2))

plt.plot(
    x,
    results2["Hybrid Throughput"],
    marker='o',
    label='Hybrid'
)

plt.plot(
    x,
    results2["DALI Throughput"],
    marker='o',
    label='DALI'
)

plt.xticks(x, results2["Batch Size"].astype(str) +
           "\n" +
           results2["Resolution"])

plt.ylabel("Images / Second")
plt.title("Hybrid vs DALI Throughput")
plt.legend()
plt.show()

print("FINAL ANALYSIS")


print("\n==================== END ====================")
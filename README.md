**人工智慧期末報告**

 >蘇家弘 廖韋瑄 紀若微
---
**簡介**
>卷積是大多數現代神經的基礎 用於計算機視覺的網絡。卷積內核是 空間不可知性和特定於通道。因此,它無法 適應不同的視覺模式 不同的空間位置。除了與位置相關的問題 卷積的接受領域在捕獲方面帶來了挑戰 遠程空間相互作用。

>為了解決上述問題,Li等。等重新考慮屬性 卷積 捲積:將卷積的連貫性轉換為視覺識別。 作者提出了“進化內核”,即特定於位置的內核和 通道不可知。由於操作的特定位置性質, 作者說自我關注屬於 捲入。

>本示例描述卷積內核,比較兩個圖像 分類模型,一個具有卷積,另一個具有 進化,還嘗試與自我關注並行 一層。
---

__設置__
```
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
import matplotlib.pyplot as plt

# Set seed for reproducibility.
tf.random.set_seed(42)
```
---
__Convolution__
>卷積仍然是計算機視覺深層神經網絡的支柱。 要了解捲積,有必要談論 卷積操作。
![image](https://github.com/gigi463682/ai_20240111/blob/e6b389f6b5df972b556366baad352dea9a6978b9/ai0111_png/MSKLsm5.png)
>考慮輸入張量 X 尺寸 H, W 和 錫。我們收集 out 卷積核 形狀 K, K, 錫。加上之間的乘法加法運算 輸入張量和我們獲得輸出張量的內核 是 與 尺寸 H, W, out。

>在上圖中 C_out=3。這使得H形的輸出張量, W和3。可以注意到,卷積核不依賴於 輸入張量的空間位置 位置不可知。另一方面,輸出中的每個通道 張量基於特定的卷積濾波器 特定於頻道。
---
__Involution__
>這個想法是要同時進行 特定位置 和 通道不可知。試圖實現這些特定屬性會帶來 挑戰。具有固定數量的內核(每個 空間位置)我們將 不 能夠處理可變分辨率 輸入張量。

>為了解決這個問題,作者考慮了 產生 每個 內核以特定的空間位置為條件。有了這個方法,我們 應該能夠輕鬆處理可變分辨率的輸入張量。 下圖提供了有關此內核生成的直覺 方法。
![image](https://github.com/gigi463682/ai_20240111/blob/22bac1690575e51f5ae51eff637d62408fe193a0/ai0111_png/jtrGGQg.png)
```
class Involution(keras.layers.Layer):
    def __init__(
        self, channel, group_number, kernel_size, stride, reduction_ratio, name
    ):
        super().__init__(name=name)

        # Initialize the parameters.
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        # Get the shape of the input.
        (_, height, width, num_channels) = input_shape

        # Scale the height and width with respect to the strides.
        height = height // self.stride
        width = width // self.stride

        # Define a layer that average pools the input tensor
        # if stride is more than 1.
        self.stride_layer = (
            keras.layers.AveragePooling2D(
                pool_size=self.stride, strides=self.stride, padding="same"
            )
            if self.stride > 1
            else tf.identity
        )
        # Define the kernel generation layer.
        self.kernel_gen = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=self.channel // self.reduction_ratio, kernel_size=1
                ),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Conv2D(
                    filters=self.kernel_size * self.kernel_size * self.group_number,
                    kernel_size=1,
                ),
            ]
        )
        # Define reshape layers
        self.kernel_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                1,
                self.group_number,
            )
        )
        self.input_patches_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                num_channels // self.group_number,
                self.group_number,
            )
        )
        self.output_reshape = keras.layers.Reshape(
            target_shape=(height, width, num_channels)
        )

    def call(self, x):
        # Generate the kernel with respect to the input tensor.
        # B, H, W, K*K*G
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)

        # reshape the kerenl
        # B, H, W, K*K, 1, G
        kernel = self.kernel_reshape(kernel)

        # Extract input patches.
        # B, H, W, K*K*C
        input_patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        # Reshape the input patches to align with later operations.
        # B, H, W, K*K, C//G, G
        input_patches = self.input_patches_reshape(input_patches)

        # Compute the multiply-add operation of kernels and patches.
        # B, H, W, K*K, C//G, G
        output = tf.multiply(kernel, input_patches)
        # B, H, W, C//G, G
        output = tf.reduce_sum(output, axis=3)

        # Reshape the output kernel.
        # B, H, W, C
        output = self.output_reshape(output)

        # Return the output tensor and the kernel.
        return output, kernel
```
---
__測試卷積層__
```
# Define the input tensor.
input_tensor = tf.random.normal((32, 256, 256, 3))

# Compute involution with stride 1.
output_tensor, _ = Involution(
    channel=3, group_number=1, kernel_size=5, stride=1, reduction_ratio=1, name="inv_1"
)(input_tensor)
print(f"with stride 1 ouput shape: {output_tensor.shape}")

# Compute involution with stride 2.
output_tensor, _ = Involution(
    channel=3, group_number=1, kernel_size=5, stride=2, reduction_ratio=1, name="inv_2"
)(input_tensor)
print(f"with stride 2 ouput shape: {output_tensor.shape}")

# Compute involution with stride 1, channel 16 and reduction ratio 2.
output_tensor, _ = Involution(
    channel=16, group_number=1, kernel_size=5, stride=1, reduction_ratio=2, name="inv_3"
)(input_tensor)
print(
    "with channel 16 and reduction ratio 2 ouput shape: {}".format(output_tensor.shape)
)
```
>結果
```
with stride 1 ouput shape: (32, 256, 256, 3)
with stride 2 ouput shape: (32, 128, 128, 3)
with channel 16 and reduction ratio 2 ouput shape: (32, 256, 256, 3)
```
---
__圖片分類__
>在本節中,我們將構建圖像分類器模型。有 是兩個模型,一個具有卷積,另一個具有卷積。

>圖像分類模型受此啟發 卷積神經網絡(CNN) Google的教程。
---
__獲取CIFAR10數據集__
```
# Load the CIFAR10 dataset.
print("loading the CIFAR10 dataset...")
(
    (train_images, train_labels),
    (
        test_images,
        test_labels,
    ),
) = keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1.
(train_images, test_images) = (train_images / 255.0, test_images / 255.0)

# Shuffle and batch the dataset.
train_ds = (
    tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    .shuffle(256)
    .batch(256)
)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(256)
```
__結果__
```
loading the CIFAR10 dataset...
```
---
__可視化數據__
```
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
```
![image]()

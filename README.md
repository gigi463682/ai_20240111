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
![image](https://github.com/gigi463682/ai_20240111/blob/2e0d20eaff57543e41416a1f9449afde9d49f0dc/ai0111_png/involution_13_0.png)
---
__卷積神經網絡__
```
# Build the conv model.
print("building the convolution model...")
conv_model = keras.Sequential(
    [
        keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding="same"),
        keras.layers.ReLU(name="relu1"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), padding="same"),
        keras.layers.ReLU(name="relu2"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), padding="same"),
        keras.layers.ReLU(name="relu3"),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10),
    ]
)

# Compile the mode with the necessary loss function and optimizer.
print("compiling the convolution model...")
conv_model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model.
print("conv model training...")
conv_hist = conv_model.fit(train_ds, epochs=20, validation_data=test_ds)
```
__結果__
```
building the convolution model...
compiling the convolution model...
conv model training...
Epoch 1/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 6s 15ms/step - accuracy: 0.3068 - loss: 1.9000 - val_accuracy: 0.4861 - val_loss: 1.4593
Epoch 2/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.5153 - loss: 1.3603 - val_accuracy: 0.5741 - val_loss: 1.1913
Epoch 3/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.5949 - loss: 1.1517 - val_accuracy: 0.6095 - val_loss: 1.0965
Epoch 4/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.6414 - loss: 1.0330 - val_accuracy: 0.6260 - val_loss: 1.0635
Epoch 5/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.6690 - loss: 0.9485 - val_accuracy: 0.6622 - val_loss: 0.9833
Epoch 6/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.6951 - loss: 0.8764 - val_accuracy: 0.6783 - val_loss: 0.9413
Epoch 7/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.7122 - loss: 0.8167 - val_accuracy: 0.6856 - val_loss: 0.9134
Epoch 8/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.7299 - loss: 0.7709 - val_accuracy: 0.7001 - val_loss: 0.8792
Epoch 9/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.7467 - loss: 0.7288 - val_accuracy: 0.6992 - val_loss: 0.8821
Epoch 10/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.7591 - loss: 0.6982 - val_accuracy: 0.7235 - val_loss: 0.8237
Epoch 11/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.7725 - loss: 0.6550 - val_accuracy: 0.7115 - val_loss: 0.8521
Epoch 12/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.7808 - loss: 0.6302 - val_accuracy: 0.7051 - val_loss: 0.8823
Epoch 13/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.7860 - loss: 0.6101 - val_accuracy: 0.7122 - val_loss: 0.8635
Epoch 14/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.7998 - loss: 0.5786 - val_accuracy: 0.7214 - val_loss: 0.8348
Epoch 15/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.8117 - loss: 0.5473 - val_accuracy: 0.7139 - val_loss: 0.8835
Epoch 16/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.8168 - loss: 0.5267 - val_accuracy: 0.7155 - val_loss: 0.8840
Epoch 17/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.8266 - loss: 0.5022 - val_accuracy: 0.7239 - val_loss: 0.8576
Epoch 18/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.8374 - loss: 0.4750 - val_accuracy: 0.7262 - val_loss: 0.8756
Epoch 19/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.8452 - loss: 0.4505 - val_accuracy: 0.7235 - val_loss: 0.9049
Epoch 20/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.8531 - loss: 0.4283 - val_accuracy: 0.7304 - val_loss: 0.8962
```
---
__捲積神經網絡__
```
# Build the involution model.
print("building the involution model...")

inputs = keras.Input(shape=(32, 32, 3))
x, _ = Involution(
    channel=3, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name="inv_1"
)(inputs)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x, _ = Involution(
    channel=3, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name="inv_2"
)(x)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x, _ = Involution(
    channel=3, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name="inv_3"
)(x)
x = keras.layers.ReLU()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10)(x)

inv_model = keras.Model(inputs=[inputs], outputs=[outputs], name="inv_model")

# Compile the mode with the necessary loss function and optimizer.
print("compiling the involution model...")
inv_model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# train the model
print("inv model training...")
inv_hist = inv_model.fit(train_ds, epochs=20, validation_data=test_ds)
```
__結果__
```
building the involution model...
compiling the involution model...
inv model training...
Epoch 1/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 9s 25ms/step - accuracy: 0.1369 - loss: 2.2728 - val_accuracy: 0.2716 - val_loss: 2.1041
Epoch 2/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.2922 - loss: 1.9489 - val_accuracy: 0.3478 - val_loss: 1.8275
Epoch 3/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.3477 - loss: 1.8098 - val_accuracy: 0.3782 - val_loss: 1.7435
Epoch 4/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.3741 - loss: 1.7420 - val_accuracy: 0.3901 - val_loss: 1.6943
Epoch 5/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.3931 - loss: 1.6942 - val_accuracy: 0.4007 - val_loss: 1.6639
Epoch 6/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.4057 - loss: 1.6622 - val_accuracy: 0.4108 - val_loss: 1.6494
Epoch 7/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.4134 - loss: 1.6374 - val_accuracy: 0.4202 - val_loss: 1.6363
Epoch 8/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.4200 - loss: 1.6166 - val_accuracy: 0.4312 - val_loss: 1.6062
Epoch 9/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.4286 - loss: 1.5949 - val_accuracy: 0.4316 - val_loss: 1.6018
Epoch 10/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.4346 - loss: 1.5794 - val_accuracy: 0.4346 - val_loss: 1.5963
Epoch 11/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.4395 - loss: 1.5641 - val_accuracy: 0.4388 - val_loss: 1.5831
Epoch 12/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.4445 - loss: 1.5502 - val_accuracy: 0.4443 - val_loss: 1.5826
Epoch 13/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.4493 - loss: 1.5391 - val_accuracy: 0.4497 - val_loss: 1.5574
Epoch 14/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.4528 - loss: 1.5255 - val_accuracy: 0.4547 - val_loss: 1.5433
Epoch 15/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.4575 - loss: 1.5148 - val_accuracy: 0.4548 - val_loss: 1.5438
Epoch 16/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.4599 - loss: 1.5072 - val_accuracy: 0.4581 - val_loss: 1.5323
Epoch 17/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.4664 - loss: 1.4957 - val_accuracy: 0.4598 - val_loss: 1.5321
Epoch 18/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.4701 - loss: 1.4863 - val_accuracy: 0.4575 - val_loss: 1.5302
Epoch 19/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.4737 - loss: 1.4790 - val_accuracy: 0.4676 - val_loss: 1.5233
Epoch 20/20
 196/196 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.4771 - loss: 1.4740 - val_accuracy: 0.4719 - val_loss: 1.5096
```
---
__比較__
>在本節中,我們將同時查看兩個模型並比較 幾點。
__參數__
>可以看到,使用類似的體系結構,CNN中的參數 比INN(進化神經網絡)大得多。
```
conv_model.summary()

inv_model.summary()
```
>Model: "sequential_3"
>┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
>┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
>┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
>│ conv2d_6 (Conv2D)               │ (None, 32, 32, 32)        │        896 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ relu1 (ReLU)                    │ (None, 32, 32, 32)        │          0 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ max_pooling2d (MaxPooling2D)    │ (None, 16, 16, 32)        │          0 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ conv2d_7 (Conv2D)               │ (None, 16, 16, 64)        │     18,496 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ relu2 (ReLU)                    │ (None, 16, 16, 64)        │          0 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ max_pooling2d_1 (MaxPooling2D)  │ (None, 8, 8, 64)          │          0 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ conv2d_8 (Conv2D)               │ (None, 8, 8, 64)          │     36,928 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ relu3 (ReLU)                    │ (None, 8, 8, 64)          │          0 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ flatten (Flatten)               │ (None, 4096)              │          0 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ dense (Dense)                   │ (None, 64)                │    262,208 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ dense_1 (Dense)                 │ (None, 10)                │        650 │
>└─────────────────────────────────┴───────────────────────────┴────────────┘
> Total params: 957,536 (3.65 MB)
> Trainable params: 319,178 (1.22 MB)
> Non-trainable params: 0 (0.00 B)
> Optimizer params: 638,358 (2.44 MB)
>Model: "inv_model"
>┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
>┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
>┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
>│ input_layer_4 (InputLayer)      │ (None, 32, 32, 3)         │          0 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ inv_1 (Involution)              │ [(None, 32, 32, 3),       │         26 │
>│                                 │ (None, 32, 32, 9, 1, 1)]  │            │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ re_lu_4 (ReLU)                  │ (None, 32, 32, 3)         │          0 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ max_pooling2d_2 (MaxPooling2D)  │ (None, 16, 16, 3)         │          0 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ inv_2 (Involution)              │ [(None, 16, 16, 3),       │         26 │
>│                                 │ (None, 16, 16, 9, 1, 1)]  │            │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ re_lu_6 (ReLU)                  │ (None, 16, 16, 3)         │          0 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ max_pooling2d_3 (MaxPooling2D)  │ (None, 8, 8, 3)           │          0 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ inv_3 (Involution)              │ [(None, 8, 8, 3), (None,  │         26 │
>│                                 │ 8, 8, 9, 1, 1)]           │            │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ re_lu_8 (ReLU)                  │ (None, 8, 8, 3)           │          0 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ flatten_1 (Flatten)             │ (None, 192)               │          0 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ dense_2 (Dense)                 │ (None, 64)                │     12,352 │
>├─────────────────────────────────┼───────────────────────────┼────────────┤
>│ dense_3 (Dense)                 │ (None, 10)                │        650 │
>└─────────────────────────────────┴───────────────────────────┴────────────┘
> Total params: 39,230 (153.25 KB)
> Trainable params: 13,074 (51.07 KB)
> Non-trainable params: 6 (24.00 B)
> Optimizer params: 26,150 (102.15 KB)

__損失和準確性圖__
>在這裡,損失和準確性圖表明INN緩慢 學習者(參數較低)。
```
plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
plt.title("Convolution Loss")
plt.plot(conv_hist.history["loss"], label="loss")
plt.plot(conv_hist.history["val_loss"], label="val_loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Involution Loss")
plt.plot(inv_hist.history["loss"], label="loss")
plt.plot(inv_hist.history["val_loss"], label="val_loss")
plt.legend()

plt.show()

plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
plt.title("Convolution Accuracy")
plt.plot(conv_hist.history["accuracy"], label="accuracy")
plt.plot(conv_hist.history["val_accuracy"], label="val_accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Involution Accuracy")
plt.plot(inv_hist.history["accuracy"], label="accuracy")
plt.plot(inv_hist.history["val_accuracy"], label="val_accuracy")
plt.legend()

plt.show()
```
![image](https://github.com/gigi463682/ai_20240111/blob/7b0ee9fde2daa5e24ff0e1af403ee2e7efa84b34/ai0111_png/involution_22_1.png)
---
__可視化卷積內核__
>為了可視化內核,我們取 K × K 每個值 捲積核。不同空間的所有代表 位置構成相應的熱圖。

作者提到:

"“我們提議的整合讓人想起自我關注和 本質上可以成為它的通用版本。”."

通過內核的可視化,我們確實可以獲得關注 圖片地圖。學習的整合內核引起人們的注意 輸入張量的單個空間位置。The 特定位置 屬性使捲入成為模型的通用空間 自我關注屬於其中。

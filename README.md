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
![image]()

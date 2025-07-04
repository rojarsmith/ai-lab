# 對數機率（log probability）

## ✅ 一句話說明：

> **對數機率 = 機率的 log（通常是 ln，也就是 log base e）**，它的目的是讓「很多很小的機率相乘」可以變成「可以加總的數字」，而且更容易計算。

------

## 🧠 為什麼要用 log？

在機器學習／機率論中，有很多情況會要「把多個機率相乘」。

例如：
 預測某一組句子出現的機率：

```
P(我, 是, 學生) = P(我) × P(是|我) × P(學生|我 是)
```

這些機率通常都很小，例如 0.001 × 0.002 × 0.05 ≈ 超級接近 0
 數值容易變得太小，無法計算，會 **underflow（數值精度問題）**

------

### ✅ 解法：取對數（log）

- **把乘法變成加法**
- 更穩定、更容易微分

這是因為：

```
log(a × b × c) = log(a) + log(b) + log(c)
```

------

## 📊 對數機率的數值範圍

| 原本的機率 p | log(p)   | 解釋                   |
| ------------ | -------- | ---------------------- |
| 1            | log(1)=0 | 最有信心的事情         |
| 0.9          | ≈ -0.1   | 高信心，損失小         |
| 0.5          | ≈ -0.69  | 模糊一半一半           |
| 0.01         | ≈ -4.6   | 很不可能的事，log 很小 |
| 0            | -∞       | 完全不可能（會報錯）   |

這表示：

> **機率越小，對數機率就越負；機率越接近 1，對數越接近 0**

------

## 🧠 對模型訓練的意義是什麼？

在機器學習裡，我們常做「最大化對數機率」（Maximum Log-Likelihood）：

- 預測越接近真實機率 → log 機率越大（越接近 0）
- 預測錯得離譜 → log 機率越小（越負）
- 所以：**我們讓模型最大化 log 機率（= 預測準） → 等同最小化 loss**

------

## ✅ 小結表格

| 名詞             | 白話解釋                                      |
| ---------------- | --------------------------------------------- |
| 機率 probability | 事情發生的可能性（0~1）                       |
| 對數機率 log(p)  | 把機率轉成「可加總」的形式（會是負數）        |
| 為什麼要用 log？ | 防止乘法過小、數值更穩定、計算容易            |
| 和 loss 有關嗎？ | 有！Binary Cross Entropy 就是基於 log(p) 來做 |


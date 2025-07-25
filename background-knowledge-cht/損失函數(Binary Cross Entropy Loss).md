# 損失函數(Binary Cross Entropy Loss)

`binary_cross_entropy`（BCE）這個公式其實是從 **機率統計中的極大似然估計（Maximum Likelihood Estimation, MLE）** 推導出來的。以下是用白話方式，帶你一步一步了解它是怎麼來的。

------

## ✅ 一句話結論：

> BCE 是根據「你預測一件事發生的機率」與「實際有沒有發生」之間的誤差來設計的，它是最大化「正確標籤發生機率」的對數。

------

## 🧠 假設情境：你在做二分類（0 或 1）

你要預測某件事會不會發生，例如：

- 電影會不會被喜歡（1：喜歡，0：不喜歡）
- 郵件是不是垃圾（1：是，0：否）

你用神經網路預測後，會得到一個機率 `p`，表示「是的機率」。

------

## 🎯 目標：讓模型預測「越準」→ 機率 `p` 越接近實際結果 `y`

### 假設你輸出的是：

```text
y = 真實標籤（0 或 1）
p = 模型預測是 1 的機率（介於 0 ~ 1）
```

那我們希望：

- 當 `y = 1`，就希望 `p` 越接近 1
- 當 `y = 0`，就希望 `p` 越接近 0

------

## 🧮 數學推導：從機率出發

在機率論中：

- 如果一個事件發生機率是 `p`，則它發生的對數機率是 `log(p)`
- 用極大似然估計（MLE）去找最佳參數時，我們會「最大化 log 機率」

對二分類，我們希望最大化：

```text
P(y=1) = p
P(y=0) = 1 - p
```

合併成：

```
P(y) = p^y * (1 - p)^(1 - y)
```

取負的對數（因為我們是最小化 loss）就變成：

```
BCE = - [ y * log(p) + (1 - y) * log(1 - p) ]
```

這就是 PyTorch `F.binary_cross_entropy()` 用的公式！

------

## 📊 對應狀況分析

| 真實 y | 預測機率 p | BCE 損失                          |
| ------ | ---------- | --------------------------------- |
| 1      | 0.99       | 很小，因為 log(0.99) 很接近 0     |
| 1      | 0.01       | 很大，因為 log(0.01) 是負很多的數 |
| 0      | 0.01       | 很小，因為 log(1 - 0.01) ≈ 0      |
| 0      | 0.99       | 很大，因為 log(1 - 0.99) 很小     |

------

## ✅ 結論總整理

| 項目                  | 說明                                         |
| --------------------- | -------------------------------------------- |
| 來源                  | 最大化機率的對數（MLE）                      |
| 設計目標              | 預測越準，loss 越小                          |
| 適用於                | 只有 0 / 1 的分類任務                        |
| 數學公式              | `- (y·log(p) + (1−y)·log(1−p))`              |
| 在 PyTorch 的對應函數 | `torch.nn.functional.binary_cross_entropy()` |




# 捷徑連接(Shortcut Connections)

------

# 一句話版本

**Shortcut connection（捷徑連線）就是讓資料「不用繞完整個神經網路」，可以直接跳過幾層，走一條快速通道。**

就像高速公路的「高架橋」，不用每個紅綠燈都停。

------

# 為什麼需要 shortcut？

想像你在訓練一個超深的神經網路（LLM 通常幾十層到上百層 Transformer）。

如果沒有 shortcut：

```
輸入 → 第1層 → 第2層 → 第3層 → ... → 第100層 → 輸出
```

問題會出現：

1. 訊息在傳遞中會「變淡」（梯度消失）
2. 越深越難訓練
3. 有些資訊其實不需要被改變那麼多

------

# Shortcut 在做什麼？

它會變成這樣：

```
輸入 x
   ↓
某一層的運算 F(x)
   ↓
把原本的 x 加回來
```

數學上其實很簡單：

```
Output = F(x) + x
```

意思是：

👉「我學新的東西，但我保留原本的資訊」

------

# 白話比喻

### 🎓 比喻 1：改作文

老師幫你改作文：

- F(x) = 老師的修改建議
- x = 你原本寫的文章

最後結果是：

```
你的文章 + 老師的修改
```

而不是整篇被覆蓋掉。

------

### 🚀 比喻 2：公司開會

- 原始企劃 = x
- 部門主管意見 = F(x)

最後決策 = 原企劃 + 修正意見

這樣就不會每一層主管都把前面全部推翻。

------

# 在 LLM 裡面長怎樣？

在 Transformer 裡，每一層都長這樣：

```
x → Attention → 加回 x
  → FeedForward → 再加回 x
```

每一個子模組後面都有 shortcut。

這叫做：

> Residual Connection（殘差連接）

因為模型學的是「差異（residual）」而不是整個東西。

------

# 為什麼這超重要？

沒有 shortcut：

- GPT 不會那麼深
- Transformer 很難訓練
- BERT 可能早就爆掉

Shortcut 讓模型：

1. 更深
2. 更穩定
3. 更容易收斂
4. 保留低層語意資訊

------

# 用一句工程師思維講

它讓模型學：

> 「如果我這層沒學到什麼，就輸出原本的 x 就好。」

換句話說：

它給了模型一個「保底機制」。

------

# 更直覺理解（非常重要）

假設某層其實不重要。

如果沒有 shortcut：

模型必須學到：

```
F(x) = x
```

但這很難。

有 shortcut：

```
Output = F(x) + x
```

只要學到：

```
F(x) = 0
```

就可以等於「這層沒影響」。

這簡單超多。

------

# 總結

Shortcut connection 的核心價值：

- 保留原始資訊
- 讓深層網路好訓練
- 讓模型學「修正量」而不是整個輸出

------

很好，這題我們用**工程師看得懂的數學**來推導。

我會分三層講：

1. 先看「沒有 shortcut」會發生什麼事
2. 再看「有 shortcut」數學怎麼變
3. 為什麼梯度會比較穩定（關鍵推導）

------

# 一、沒有 Shortcut 時會發生什麼？

假設一個很深的網路：

```math
x_{l+1} = F_l(x_l)
```

總共有 L 層。

最後 loss 是：

```math
\mathcal{L}
```

我們要算對某一層的梯度：

```math
\frac{\partial \mathcal{L}}{\partial x_l}
```

根據鏈式法則：

```math
\frac{\partial \mathcal{L}}{\partial x_l}=
\frac{\partial \mathcal{L}}{\partial x_L}
\prod_{i=l}^{L-1}
\frac{\partial x_{i+1}}{\partial x_i}
```

但

```math
\frac{\partial x_{i+1}}{\partial x_i}=F_i'(x_i)
```

所以變成：

```math
\frac{\partial \mathcal{L}}{\partial x_l}=
\frac{\partial \mathcal{L}}{\partial x_L}
\prod_{i=l}^{L-1}
F_i'(x_i)
```



------

## 問題在哪？

這是一連串「矩陣乘法」。

如果每層的梯度大小大約是：

- 0.9 → 會指數衰減
- 1.1 → 會指數爆炸

例如：

```math
0.9^{100} ≈ 0.000026
```

梯度幾乎消失。

這就是 **vanishing gradient**。

------

# 二、有 Shortcut 時數學變什麼？

現在加入 residual connection：

```math
x_{l+1} = x_l + F_l(x_l)
```

來算導數：

```math
\frac{\partial x_{l+1}}{\partial x_l}=1 + F_l'(x_l)
```

關鍵就在這個 **+1**。

------

# 三、為什麼 +1 讓梯度穩定？

現在整體梯度變成：

```math
\frac{\partial \mathcal{L}}{\partial x_l}=
\frac{\partial \mathcal{L}}{\partial x_L}
\prod_{i=l}^{L-1}
\left(1 + F_i'(x_i)\right)
```



------

## 觀察關鍵情況

假設：

```math
F_i'(x_i) ≈ 0
```

（很常發生，因為初始化很小）

那麼：

沒有 shortcut：

```math
\prod F_i' ≈ 0
```

梯度直接死掉。

------

有 shortcut：

```math
\prod (1 + 0) = 1
```

梯度完整保留下來。

這就是數學本質。

------

# 更進一步：特徵值觀點（更深一點）

在深度網路中，

梯度實際上是：

```math
\prod W_i
```

矩陣連乘。

如果矩陣特徵值 < 1
→ 指數衰減

如果 > 1
→ 爆炸

------

加入 residual：

```math
W_i + I
```

也就是：

```math
I + small_matrix
```

線性代數裡有個重要事實：

如果：

```math
∥F∥很小
```

那麼：

```math
∥I+F∥≈1
```

所以連乘後不會爆炸也不會消失。

這叫做：

> 把 transformation 從「純映射」變成「接近 identity 的微小擾動」

------

# 直覺理解（非常重要）

沒有 shortcut：

模型每一層都要「完整重建表示」。

有 shortcut：

模型只學：

```math
\text{微調}
```

就像數值方法中的：

```math
x_{t+1} = x_t + \Delta
```

這其實跟 ODE 離散化一模一樣。

ResNet 本質上像在解微分方程。

------

# 為什麼這對 Transformer 特別重要？

Transformer 很深（例如 GPT-3 96層）。

沒有 residual：

梯度 = 96 次矩陣連乘
幾乎必死。

有 residual：

每一層接近 identity mapping
整體梯度穩定在 O(1)

------

# 一句話總結數學核心

Residual 讓每層的 Jacobian：

```math
J = I + \epsilon
```

而不是：

```math
J = \epsilon
```

連乘：

```math
(I + \epsilon_1)(I + \epsilon_2)...(I + \epsilon_n)
```

這比：

```math
\epsilon_1 \epsilon_2 ... \epsilon_n
```

穩定非常多。

------

這題非常關鍵。
很多人知道「Pre-LN 比 Post-LN 穩定」，但不知道**數學上為什麼**。

我用你習慣的「推導 + 工程直覺」方式講清楚。

------

# 一、先看兩種結構差在哪

## ✅ Post-LN（原始 Transformer, 2017）

```math
x_{l+1} = \text{LN}(x_l + F(x_l))
```

流程：

```
x → F(x)
  → + x
  → LayerNorm
```

------

## ✅ Pre-LN（現在 GPT/BERT 幾乎都用）

```math
x_{l+1} = x_l + F(\text{LN}(x_l))
```

流程：

```
x → LayerNorm → F
  → + x
```

------

# 二、核心差別在哪？

關鍵在：

> **LayerNorm 在 residual 之前，還是之後**

這會直接影響「梯度通路是否乾淨」。

------

# 三、從梯度推導開始（關鍵）

我們關心：

```math
\frac{\partial \mathcal{L}}{\partial x_l}
```



------

# 四、先看 Post-LN 的梯度

Post-LN：

```math
x_{l+1} = \text{LN}(x_l + F(x_l))
```

設：

```math
y = x_l + F(x_l) \\
x_{l+1} = LN(y)
```

梯度變成：

```math
\frac{\partial x_{l+1}}{\partial x_l}=
\frac{\partial LN(y)}{\partial y}
\cdot
\left(I + F'(x_l)\right)
```

注意這個：

```math
\frac{\partial LN(y)}{\partial y}
```

LayerNorm 的 Jacobian 不是 identity。

它包含：

- 減均值
- 除標準差
- scale + shift

這會讓 Jacobian：

- 依賴整個向量
- 不是對角矩陣
- 特徵值可能 < 1

------

## 重要後果

梯度變成：

```math
\prod \left( J_{LN} (I + F') \right)
```

因為每層都會乘上一個 **LayerNorm Jacobian**。

如果：

```math
∥J_{LN}∥ < 1
```

就會逐層縮小。

------

# 五、再看 Pre-LN 的梯度

Pre-LN：

```math
x_{l+1} = x_l + F(LN(x_l))
```

求導：

```math
\frac{\partial x_{l+1}}{\partial x_l}=
I + F'(LN(x_l)) \cdot LN'(x_l)
```

關鍵來了：

這裡有一個 **裸的 I**

而且 **沒有被 LN 包住**。

------

## 梯度鏈式展開

整體梯度：

```math
\prod
\left(I + small_term\right)
```

這跟我們上一題 residual 推導一樣：

```math
(I + \epsilon_1)(I + \epsilon_2)...
```

而不是：

```math
J_{LN}(I + \epsilon)
```



------

# 六、最重要的差異（真正本質）

## Post-LN：

梯度主幹被 LN 破壞

沒有乾淨的 identity path

------

## Pre-LN：

梯度永遠有一條：

```math
\frac{\partial x_{l+1}}{\partial x_l} \approx I
```

就算 F 壞掉
就算 LN 壞掉

identity path 仍存在。

------

# 七、用線性代數觀點看

假設：

```math
∥F'∥ 很小
```



------

### Pre-LN

```math
J = I + \epsilon
```

特徵值 ≈ 1

深層連乘仍然 ≈ O(1)

------

### Post-LN

```math
J = J_{LN}(I + \epsilon)
```

而：

LayerNorm 會把向量「投影到零均值空間」

它的 Jacobian 本質上是：

```math
I - \frac{1}{d}\mathbf{1}\mathbf{1}^T
```

這個矩陣：

- 有一個 0 特徵值
- 其他是 < 1

所以：

```math
∥J_{LN}∥ ≤ 1
```

乘很多層後：

梯度被壓縮。

------

# 八、實際工程現象

### Post-LN 訓練深層模型會：

- 前幾層梯度很小
- 很難 warmup
- 需要很小 learning rate
- 超過 24 層容易不穩

------

### Pre-LN：

- 可以 96 層以上
- 不需要那麼長 warmup
- 梯度更均勻

這就是 GPT-2 之後全面轉向 Pre-LN 的原因。

------

# 九、更深一層（ODE 視角）

Pre-LN 更像：

```math
x_{l+1} = x_l + f(x_l)
```

這是顯式 Euler 積分。

穩定性條件好控制。

------

Post-LN 是：

```math
x_{l+1} = LN(x_l + f(x_l))
```

等於每一步都強制重新投影。

這破壞了動力系統的連續性。

------

# 十、結論（最核心一句）

Pre-LN 比 Post-LN 穩定的真正原因是：

> **它保留了完全乾淨的 identity gradient path**

而 Post-LN 把 identity path 包進 LayerNorm 裡面破壞掉。

------

# 十一、為什麼現在幾乎所有 LLM 都是 Pre-LN？

因為當模型變成：

- 70B 參數
- 100+ 層

Post-LN 幾乎不可訓練。

Pre-LN 是讓 scaling law 成立的關鍵工程修正之一。

------


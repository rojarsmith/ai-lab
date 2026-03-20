# AdamW

在訓練 **LLM（Large Language Model）** 時，最常用的優化器之一就是 **AdamW**。
你可以把它理解為：

> **一種「聰明調整學習速度 + 避免模型變太複雜」的梯度更新方法。**

我們先用 **白話概念**講，再給一個**具體例子**。

------

# 一、先理解：模型訓練到底在做什麼

訓練 LLM 本質是在做一件事：

**不停調整參數，讓預測更準。**

假設模型有一個權重：

```math
w
```

每次訓練會算出 **梯度（gradient）**

```math
g
```

代表：

- 如果 (g>0)：權重應該變小
- 如果 (g<0)：權重應該變大

最基本的更新方式叫 **SGD**：

```math
w = w - \eta g
```

其中：

- η = learning rate

但這方法有很多問題：

- 有些參數更新太快
- 有些更新太慢
- 梯度會震盪

所以出現 **Adam**。

------

# 二、Adam 是什麼

Adam 的想法是：

> **不要只看當前梯度，要看「歷史梯度趨勢」。**

Adam會計算兩個東西：

### 1️⃣ 梯度平均（Momentum）

```math
m_t
```

意思是：

> 梯度的「移動平均」

類似：

「最近幾步的梯度大概是什麼方向」

------

### 2️⃣ 梯度平方平均

```math
v_t
```

意思是：

> 梯度大小的平均

這可以讓：

- 大梯度 → 更新慢一點
- 小梯度 → 更新快一點

平方平均

```math
\sqrt{v_t}
```

------

最後更新公式：

```math
w = w - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
```

直觀意思：

> **方向用平均梯度，速度用梯度大小調整。**

```math
更新幅度=\frac{想走的方向}{最近梯度大小}
```

所以 Adam 比 SGD **穩定很多**。

分母的 ϵ白話就是：

> **一個很小很小的保護數字，用來避免分母變成 0，順便讓更新更穩定。**

------

# 三、Adam 的問題

Adam 有一個問題：

> **Weight decay（權重衰減）和 Adam 的更新混在一起。**

Weight decay 是什麼？

就是：

```math
w = w - \lambda w
```

意思：

> 讓權重慢慢變小，避免模型過度複雜（overfitting）

但在 **Adam 原始版本**中：

weight decay **混在梯度裡**

這會導致：

- 正則化效果變差
- 大模型效果不好

------

# 四、AdamW 的改進

AdamW 的想法很簡單：

> **把 weight decay 和梯度更新分開**

更新分兩步：

### Step1：Adam 更新

```math
w = w - \eta \frac{m_t}{\sqrt{v_t}+\epsilon}
```



### Step2：Weight Decay

```math
w = w - \eta \lambda w
```

也可以寫成：

```math
w = w - \eta \left(\frac{m_t}{\sqrt{v_t}} + \lambda w \right)
```



------

# 五、白話比喻

想像你在 **調整一台收音機旋鈕**。

模型參數 = 旋鈕位置

訓練目標 = 聽到最清楚的聲音

------

### SGD

每次只看：

> 「現在聲音變大還變小？」

然後調整。

缺點：

- 會來回震盪

------

### Adam

會記住：

- 最近調整方向
- 最近變化大小

就像：

> 「最近幾次調整好像都往左比較好」

所以更穩。

------

### AdamW

再加一條規則：

> 「不要把旋鈕轉太大」

每次都會 **稍微把旋鈕往中間拉一點**。

這就是 **weight decay**。

------

# 六、數字例子（直覺）

假設：

```
權重 w = 10
learning rate = 0.01
gradient = 2
weight decay = 0.1
```

------

### Adam更新

假設 Adam 算出更新量：

```
Δ = 0.02
```

更新後：

```
w = 10 - 0.02
w = 9.98
```

------

### AdamW 再做 weight decay

```
decay = lr * wd * w
       = 0.01 * 0.1 * 9.98
       = 0.00998
```

再更新：

```
w = 9.98 - 0.00998
w ≈ 9.970
```

所以：

**AdamW = Adam更新 + 權重縮小**

------

# 七、為什麼 LLM 幾乎都用 AdamW

原因很簡單：

1️⃣ 收斂穩定
2️⃣ 適合巨大參數量
3️⃣ weight decay 更正確
4️⃣ 大模型效果更好

所以：

- GPT
- BERT
- LLaMA
- T5

幾乎全部用 **AdamW**。

------

# 八、PyTorch 範例

實際訓練 LLM 時會看到：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
```

意思：

- lr = 學習率
- betas = 梯度平均係數
- weight_decay = 0.01

------

# 九、一句話理解 AdamW

**AdamW = Adam（自適應梯度） + 正確實作的 Weight Decay。**




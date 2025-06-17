# torch.nn.Embedding 用途解析

## 它其實跟 word2vec、skip-gram 沒什麼關係

如果你跟我一樣，是在看了 Transformer 或其他關於 word embeddings 的 paper 之後，才回頭來學 PyTorch 實作，那你一開始可能會被 `torch.nn.Embedding` 搞得一頭霧水。

查了官方文件之後，反而更疑惑：

> A simple lookup table that stores embeddings of a fixed dictionary and size.
> This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.
>
> Parameters:
>
> - num_embeddings (int) – size of the dictionary of embeddings
> - embedding_dim (int) – the size of each embedding vector
>
> ...

那到底是什麼意思？它有訓練過嗎？背後用的是 word2vec、GloVe 還是某種 pretrained embedding？

## 真相很簡單：都不是，它只是隨機初始化

其實答案非常單純——`torch.nn.Embedding` 只是建立一個**隨機初始化的查找表（lookup table）**，和任何 pretrained 的模型一點關係都沒有，預設情況下數值是常態分佈。

------

## 文件再看一次：num_embeddings 和 embedding_dim 是什麼？

舉個例子，如果你寫：

```python
nn.Embedding(3, 5)
```

這代表什麼？

- `3`：代表你的 vocabulary size（總共可以容納 3 種 token）
- `5`：代表每個 token 要被表示成一個 5 維的向量（embedding_dim）

這行程式的背後，其實就像這樣的資料結構：

```python
{
  0: [0.12, 0.56, 0.78, 0.91, 0.34], # 五個隨機的floats來代表0 這個token
  1: [0.45, 0.23, 0.67, 0.89, 0.10], # 五個隨機的floats來代表1 這個token
  2: [0.98, 0.77, 0.54, 0.22, 0.66], # 五個隨機的floats來代表2 這個token
}
```

這些數字都是隨機生成的浮點數，一開始什麼意思都沒有，會隨著訓練逐漸調整。

------

## 那文字怎麼處理？Tokenizer 在做什麼？

你可能會問：「可是我輸入的是文字，不是數字啊？」

這就牽涉到 **Tokenizer** 的角色。Tokenizer 就是把文字對應到數字的工具，例如：

```python
{'你': 0, '好': 1, '嗎': 2}
```

這樣一句「你好好嗎嗎」就會被轉換成 `[0, 1, 1, 2, 2]`

再輸入 `nn.Embedding`，就會對應出向量表達：

```python
[
  Embedding[0],
  Embedding[1],
  Embedding[1],
  Embedding[2],
  Embedding[2],
]
```

實際上就是拿 token 的 index 去查表。

------

## 範例：從文字到嵌入向量

以句子「你好好嗎嗎」為例，經過 tokenizer → embedding：

```text
Input tokens: [0, 1, 1, 2, 2]
Output embeddings:
[
  [0.12, 0.56, 0.78, 0.91, 0.34],
  [0.45, 0.23, 0.67, 0.89, 0.10],
  [0.45, 0.23, 0.67, 0.89, 0.10],
  [0.98, 0.77, 0.54, 0.22, 0.66],
  [0.98, 0.77, 0.54, 0.22, 0.66],
]
```

句子「你嗎嗎好」則是 `[0, 2, 2, 1]`，查表後得到另一組對應向量。

------

## 那這些向量有用嗎？怎麼訓練？

目前這些向量還沒有任何語意，是隨機的。但接下來你可以接上一個分類任務，像是判斷句子是否是髒話：

```python
"你好好嗎嗎" → label 0（不是髒話）  
"你嗎嗎好" → label 1（是髒話）
```

這時候只要你沒有凍結 `nn.Embedding`（即它仍可訓練），那麼這些隨機初始化的向量，就會根據 loss 逐步被調整，最後變成語意上有意義的表達。

------

## 為什麼 vocabulary size 要設很大？

你在定義 `nn.Embedding(num_embeddings, embedding_dim)` 時，第一個參數就是你的「字典大小」。

像我們剛剛設的是 3，那就只能處理 index 為 0、1、2 的 token。只要傳進去的 index 超過 2，就會出錯（index out of range）。

實際上，語言模型通常會設定一個夠大的值，例如 80,000，來搭配 tokenizer 處理自然語言資料。

------

## 小結

- `nn.Embedding` 是一個**可訓練的查找表**，裡面每個詞的向量一開始都是隨機的。
- 它本身跟 word2vec、GloVe、skip-gram 都無關。
- Tokenizer 負責把文字轉換成數字 index，embedding 則把 index 轉成向量。
- 這些向量會在後續訓練中根據任務目標（分類、翻譯、情緒分析等）被優化。

------

如果你是第一次接觸，確實很容易被名字搞混，但掌握住「它就是一張隨機初始化、可訓練的對照表」這個核心概念，就容易很多了！

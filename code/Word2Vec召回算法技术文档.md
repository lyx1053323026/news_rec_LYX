# Word2Vec 向量召回算法技术文档

## 一、算法概述

Word2Vec 召回是一种基于**文章语义向量**的 i2i 召回方法。与 ItemCF 和 Swing 等基于用户行为共现的方法不同，Word2Vec 召回将用户点击序列视为"句子"，将每篇文章视为"词"，通过 Skip-gram 模型学习文章的稠密向量表示，再用向量相似度进行召回。

**核心思想：** 被同一用户在同一会话中连续点击的文章，在向量空间中应当距离相近。这是一种**序列上下文**建模，区别于共现统计建模。

## 二、算法原理

### 2.1 Word2Vec Skip-gram 模型

Skip-gram 的目标是：给定中心词（当前点击的文章），预测上下文词（同一用户点击的前后文章）。

$$\mathcal{L} = \sum_{u} \sum_{t=1}^{L_u} \sum_{-w \leq j \leq w, j \neq 0} \log P(a_{t+j} \mid a_t)$$

其中：
- $a_t$：用户 $u$ 在位置 $t$ 点击的文章
- $w$：窗口大小（本方案 `window=3`）
- $L_u$：用户 $u$ 的点击序列长度

条件概率通过负采样近似：

$$P(a_O \mid a_I) = \sigma(\mathbf{v}_{a_O}^\top \mathbf{v}_{a_I}) \prod_{k=1}^{K} \sigma(-\mathbf{v}_{a_k}^\top \mathbf{v}_{a_I})$$

### 2.2 训练参数

| 参数 | 值 | 说明 |
|------|:---:|------|
| `vector_size` | 256 | 文章向量维度 |
| `window` | 3 | 上下文窗口大小 |
| `sg` | 1 | Skip-gram（而非 CBOW） |
| `hs` | 0 | 使用负采样（而非 Hierarchical Softmax） |
| `negative` | 5 | 每个正样本配 5 个负样本 |
| `min_count` | 1 | 所有文章都保留（无低频过滤） |
| `epochs` | 1 | 训练轮数 |
| `workers` | 10 | 并行训练线程数 |

**参数选择依据：**

- **Skip-gram vs CBOW**：新闻推荐中每篇文章都是长尾分布，Skip-gram 对低频文章效果更好（每个样本独立更新），CBOW 会平均上下文向量导致低频文章信息被稀释
- **window=3**：新闻浏览场景中，用户在短时间内（约 3 篇以内）点击的文章具有较强主题关联；更大的窗口会引入噪声
- **negative=5**：小数据集（约 34K 篇文章）下 5 个负样本足够，更多负样本不会显著提升质量但增加训练时间
- **epochs=1**：配合窗口=3 和负采样，单轮训练已能学到合理的表示；多轮训练在行为序列数据上容易过拟合

### 2.3 向量相似度

召回和排序阶段使用**余弦相似度**衡量文章间的语义关联：

$$\cos(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}| \cdot |\vec{b}|}$$

值域为 $[-1, 1]$，1 表示方向完全相同，0 表示正交（无关联），-1 表示完全相反。

---

## 三、工程实现

### 3.1 训练流程

```
用户点击日志 (click.pkl)
    │
    ▼
按 user_id 聚合点击序列
    {user1: [article_1, article_5, ..., article_n]}
    │
    ▼
将 article_id 转为字符串（Gensim 要求）
    [["1", "5", ...], ["2", "8", ...], ...]
    │
    ▼
Word2Vec(sg=1, vector_size=256, window=3, ...)
    │
    ├── 已缓存 w2v.m → 直接加载
    └── 否则 → 训练后保存
    │
    ▼
article_vec_map: {article_id: np.ndarray(256,)}
```

### 3.2 向量索引

原始代码使用 **Annoy 兼容接口 + FAISS 底层实现**：

| 组件 | 说明 |
|------|------|
| 索引类型 | `IndexFlatIP`（内积索引） |
| 归一化 | L2 normalize → 内积 = 余弦相似度 |
| 向量维度 | 256 |
| 搜索算法 | 暴力搜索（Flat，无近似） |

**为什么用 Flat（暴力）而非近似索引？**
- 文章总数约 34K，256 维向量的暴力搜索成本极低（毫秒级）
- Flat 索引保证精确结果，无需为速度牺牲精度
- `AnnoyIndex.build(100)` 的参数 100 在 FAISS Flat 索引中无实际作用，仅保留接口兼容性

### 3.3 召回策略

与 ItemCF 和 Swing 使用**多个种子物品**不同，Word2Vec 召回极其简洁——**仅使用用户最后一次点击的文章作为种子**：

```python
interacted_items = user_item_dict[user_id]
interacted_items = interacted_items[-1:]  # 只取最后 1 个
```

对种子文章：
1. 取出其 256 维向量
2. 在 FAISS 索引中搜索 top-100 最近邻
3. 距离转换：`sim_score = 2 - distance`（FAISS angular 距离 ∈ [0, 2]，转换为相似度 ∈ [0, 2]）
4. 排除已交互的文章，取 top-100

**设计考量：**

| 对比维度 | Word2Vec | ItemCF | Swing |
|---------|:--------:|:------:|:-----:|
| 种子物品数 | **1**（最后一次） | 全部历史 | 最近 2 次 |
| 候选来源 | 向量近邻 | 共现矩阵 | 共现矩阵 |
| 位置衰减 | 无 | 有（0.9^loc） | 有（0.7^loc） |
| 召回数 | top-100 | top-100 | top-100 |

Word2Vec 只用 1 个种子物品，因为向量空间已经编码了语义关系，多个种子的加权累加反而可能稀释信号。相比之下，ItemCF 等基于共现的方法需要多个种子来覆盖更多候选。

### 3.4 排序阶段特征

Word2Vec 召回结果为排序模型提供了 2 个特征：

| 特征 | 函数 | 公式 |
|------|------|------|
| `user_last_click_article_w2v_sim` | `func_w2w_last_sim` | $\cos(\vec{v}_{last}, \vec{v}_{candidate})$ |
| `user_click_article_w2w_sim_sum_2` | `func_w2w_sum(num=2)` | $\sum_{k=1}^{2} \cos(\vec{v}_{I_k}, \vec{v}_{candidate})$ |

其中 `func_w2w_sum_2` 使用了用户的**最近 2 次**点击文章（而非召回的仅 1 次），是排序阶段的更细粒度特征。

---

## 四、与其他召回方式对比

| 维度 | Word2Vec | ItemCF (v2) | Swing |
|------|:--------:|:-----------:|:-----:|
| **理论基础** | 分布式语义假设 | 协同过滤 | 二部图共现惩罚 |
| **输入** | 用户序列（文章 ID 序列） | 用户-物品交互矩阵 | 用户-物品交互矩阵 |
| **输出** | 稠密向量（256 维） | 稀疏相似矩阵（dict of dict） | 稀疏相似矩阵（dict of dict） |
| **时序建模** | ✅ 窗口内顺序 | ✅ 三项时间权重 | ❌ |
| **冷启动** | ✅ 新文章有向量即可召回 | ❌ 无交互则无相似度 | ❌ 同左 |
| **可解释性** | 低（黑盒向量） | 高（可追溯共现对） | 中（可追溯用户对） |
| **计算复杂度** | 低（向量近邻 O(Nd)） | 中（二重循环遍历） | 高（O(n²) 用户对） |
| **存储开销** | ~35MB（34K×256×4B） | ~27MB（稀疏 dict） | ~27MB（稀疏 dict） |
| **召回信号** | 语义关联 | 行为共现 | 独立共现 |

**Word2Vec 的核心优势：**
1. **语义泛化**：即使两篇文章从不同用户共现过，只要它们在序列中的上下文模式相似，就能学到相近的向量
2. **增量友好**：新文章只需一次前向传播即可得到向量，无需全量重算相似矩阵
3. **互补信号**：与 ItemCF / Swing 的共现信号正交，融合后能覆盖更多样的用户兴趣

**Word2Vec 的局限：**
1. 仅用最后 1 次点击作为种子，召回覆盖的多样性可能不足
2. 向量维度固定（256），无法自适应调整表达能力
3. 对点击序列长度敏感——短序列用户（1-2 次点击）的向量质量受影响

---

## 五、代码结构

```
recall_w2v.py
├── AnnoyIndex                        # FAISS 兼容封装
│   ├── __init__(dim, metric)         # 初始化索引参数
│   ├── add_item(id, vector)          # 添加向量
│   ├── build(n_trees)               # 构建索引（L2 归一化 + Flat IP）
│   └── get_nns_by_vector(vec, n)     # 查询 top-n 近邻
├── word2vec(df, f1, f2, model_path)  # Word2Vec 训练
│   ├── groupby 聚合用户点击序列
│   ├── 加载缓存 or 训练新模型
│   ├── Skip-gram, 256d, window=3
│   └── 返回 {article_id: np.ndarray}
├── recall() [@multitasking.task]      # 多线程召回
│   ├── 取用户最后 1 次交互作为种子
│   ├── FAISS 查询 top-100 最近邻
│   ├── 距离转相似度: 2 - distance
│   └── label 标记（验证模式）
└── __main__                           # 主流程
    ├── word2vec 训练 → article_vec_map
    ├── FAISS 构建索引
    ├── 多线程召回 → 合并 → 排序
    ├── 计算 HR@K / MRR@K
    └── 保存 recall_w2v.pkl
```

**依赖关系：**
- 输入：`click.pkl`（用户历史点击）、`query.pkl`（验证/测试用户）
- 中间产物：`w2v.m`（Gensim 模型）、`article_w2v.pkl`（向量映射）
- 输出：`recall_w2v.pkl`（召回结果）
- 融合权重：1.0（与 ItemCF、Swing 同级）

---

## 六、使用方式

```bash
# 验证模式（计算指标）
python recall_w2v.py --mode valid --logfile "w2v.log"

# 在线模式（生成最终召回结果）
python recall_w2v.py --mode online --logfile "w2v_online.log"
```

在完整流水线中的位置（`test.sh`）：

```
data.py → recall_itemcf.py → recall_swing.py → recall_w2v.py → recall.py → rank_feature.py → rank_lgb.py
```

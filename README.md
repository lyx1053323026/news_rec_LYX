# 天池新闻推荐竞赛方案

> 零基础入门推荐系统 - 新闻推荐
>
> 比赛地址：[天池大赛 - 新闻推荐](https://tianchi.aliyun.com/competition/entrance/531842/introduction)

## 赛题简介

根据用户的历史点击日志，预测每个用户下一次会点击哪篇新闻，最终为每个用户推荐 5 篇文章。

| 项目 | 说明 |
|------|------|
| 数据量 | 30 万用户，近 300 万次点击，36 万+ 篇新闻 |
| 训练集 | 20 万用户点击日志 |
| 测试集 | A/B 各 5 万用户 |
| 评估指标 | **HitRate@5**、**MRR@5** |

## 方案架构

采用经典的「**召回 + 排序**」两阶段架构：

```
                ┌─────────────────┐
                │   用户点击日志    │
                └────────┬────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐   ┌──────▼──────┐   ┌────▼────┐
    │ ItemCF  │   │   Swing     │   │ Word2Vec│
    └────┬────┘   └──────┬──────┘   └────┬────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                  ┌──────▼──────┐
                  │  多路召回合并 │
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │  特征工程    │
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │ LightGBM    │
                  │   排序      │
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │  Top 5 推荐 │
                  └─────────────┘
```

## 召回策略

采用 **3 路多路召回**，通过不同策略的差异性提高整体召回覆盖率：

### 1. ItemCF 协同过滤 (`recall_itemcf.py`)

基于物品的协同过滤，使用共现次数 + 位置权重 + 时间衰减计算物品相似度。

### 2. Swing 协同过滤 (`recall_swing.py`)

通过引入「用户对共现惩罚」机制，抑制高重叠度用户对带来的噪声信号，更准确地捕捉物品间的独立共现关系。

$$\text{sim}(i, j) = \sum_{u \in U_i \cap U_j} \sum_{\substack{v \in U_i \cap U_j \\ v \neq u}} \frac{1}{\alpha + |I_u \cap I_v|}$$

- 用户历史截断：最近 200 条
- 热门物品用户采样：最多 300 个用户
- 单路验证集 HitRate@5: **28.76%**

### 3. Word2Vec 向量召回 (`recall_w2v.py`)

基于 Word2Vec 训练文章向量，通过余弦相似度计算物品间的语义关联，进行 i2i 召回。

### 召回合并 (`recall.py`)

- 各路召回结果经 MMS 归一化后加权合并
- 当前融合权重：ItemCF=1, Swing=1, Word2Vec=1
- 去重、保留有正样本的用户

## 排序模型

### 特征工程 (`rank_feature.py`)

构造 **30+ 维排序特征**，包括：

| 类别 | 特征 |
|------|------|
| 文章属性 | 类目 ID、创建时间、字数 |
| 用户历史统计 | 点击时间差、文章创建时间差、字数统计、点击时段 |
| 计数特征 | 用户点击数、文章被点击数、用户类目点击数 |
| 召回相似度 | 3 路召回的相似度加权得分（最强信号） |

### LightGBM 排序 (`rank_lgb.py`)

```python
LGBMClassifier(
    num_leaves=64,
    max_depth=10,
    learning_rate=0.05,
    n_estimators=10000,    # 配合 early_stopping
    subsample=0.8,
    feature_fraction=0.8,
    reg_alpha=0.5,         # L1 正则
    reg_lambda=0.5,        # L2 正则
)
```

- **GroupKFold 5 折交叉验证**（按 user_id 分组，防止数据泄露）
- Early stopping（验证集 AUC 连续 100 轮不提升则停止）
- 测试集预测：5 折概率取均值

## 项目结构

```
.
├── README.md                      # 本文件
├── .gitignore
├── tcdata/
│   └── node.txt                   # 数据集字段说明
└── 源码/tianchi-news-recommendation-master/
    ├── README.md
    ├── requirements.txt            # Python 依赖
    └── code/
        ├── data.py                 # 数据预处理与验证集划分
        ├── utils.py                # 工具函数（日志、评估、提交生成）
        ├── recall_itemcf.py        # ItemCF 召回
        ├── recall_swing.py         # Swing 召回 ⭐
        ├── recall_w2v.py           # Word2Vec 召回
        ├── recall.py               # 多路召回合并
        ├── rank_feature.py         # 排序特征工程
        ├── rank_lgb.py             # LightGBM 排序模型
        ├── test.sh                 # 一键运行脚本
        ├── ItemCF算法技术文档.md     # 改进 ItemCF 算法详细文档
        ├── Swing算法技术文档.md     # Swing 算法详细文档
        └── 排序模型技术文档.md      # 排序模型详细文档
```

## 快速开始

### 环境要求

- Python 3.13+
- 依赖见 `requirements.txt`

### 数据准备

从比赛官网下载以下数据文件，放入 `tcdata/` 目录：

| 文件 | 说明 |
|------|------|
| `train_click_log.csv` | 训练集用户点击日志 |
| `testA_click_log.csv` | 测试集 A 用户点击日志 |
| `articles.csv` | 新闻文章信息表 |
| `articles_emb.csv` | 新闻文章 Embedding 向量 |
| `sample_submit.csv` | 提交样例文件 |

### 一键运行

```bash
cd 源码/tianchi-news-recommendation-master
pip install -r requirements.txt
cd code
bash test.sh
```

### 分步运行

```bash
cd code

# 1. 数据预处理
python data.py --mode valid --logfile "run.log"

# 2. 多路召回
python recall_itemcf.py --mode valid --logfile "run.log"
python recall_swing.py --mode valid --logfile "run.log"
python recall_w2v.py --mode valid --logfile "run.log"

# 3. 召回合并
python recall.py --mode valid --logfile "run.log"

# 4. 排序
python rank_feature.py --mode valid --logfile "run.log"
python rank_lgb.py --mode valid --logfile "run.log"
```

## 离线验证体系

从训练集用户中随机采样 **5 万个用户**，将每个用户的最后一条点击记录剔除作为验证集 ground truth，剩余的历史点击与测试集点击合并作为总历史记录。这套验证方式保证了线下指标与线上指标同增同减，差距可控。

## 技术文档

- [改进 ItemCF 协同过滤算法技术文档](源码/tianchi-news-recommendation-master/code/ItemCF算法技术文档.md)
- [Swing 协同过滤算法技术文档](源码/tianchi-news-recommendation-master/code/Swing算法技术文档.md)
- [排序模型技术文档](源码/tianchi-news-recommendation-master/code/排序模型技术文档.md)

## License

仅供学习研究使用。

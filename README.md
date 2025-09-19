# DualHCL：Dual Hypergraph Contrastive Learning for Network Alignment

本项目是 DualHCL 算法的实现

#### 环境依赖

  - `python==3.9.12`
  - `networkx==2.6.3`
  - `numpy==1.22.4`
  - `pytorch==1.12.1`
  - `scikit-learn==1.5.2`
  - `torch-geometric ==2.6.1`
  - `gensim==4.3.3`

#### 项目结构

```
├── data/                    # 数据集目录
│   ├── douban/              # Douban 在线-离线网络数据集
│   ├── dblp/                # DBLP 数据集
│   └── ...
├── DualHCL.py               # 主算法实现
├── run_DualHCL.sh           # 运行示例脚本
├── utils.py                 # 工具函数
├── matcher.py               # 评估指标计算
└── node2vec.py              # 随机游走实现
```

#### 如何运行

你可以通过以下两种方式运行代码：

**1. 使用 Shell 脚本 (推荐)**

脚本 `run_DualHCL.sh` 提供了在 Douban 数据集上运行的示例。你可以直接运行此脚本：

```bash
run_DualHCL.sh
```

你也可以修改 `run_DualHCL.sh` 中的变量来切换不同的数据集，例如：

  - `PD=data/twitter_youtube`
  - `PD=data/twitter_foursquare`
  - `PD=data/dblp`
  - `PD=data/wd`

**2. 使用 Python 命令**

你也可以直接使用 `python` 命令来运行，这提供了更高的灵活性。例如，要在 Douban 数据集上运行 DualHCL 算法，训练集比例为0.8，请在项目根目录中运行以下命令：

```bash
python DualHCL.py \
    --s_edge data/douban/online.txt \
    --t_edge data/douban/offline.txt \
    --gt_path data/douban/node,split=0.8.test.dict \
    --train_path data/douban/node,split=0.8.train.dict
```

#### 参数说明

以下是 `DualHCL.py` 脚本中的主要参数说明：

  - `--s_edge`: 源网络边文件路径。
  - `--t_edge`: 目标网络边文件路径。
  - `--gt_path`: 测试集真实匹配节点字典文件路径。
  - `--train_path`: 训练集（锚节点）字典文件路径。
  - `--dim`: 节点嵌入的维度 (默认: 128)。
  - `--lr`: 学习率 (默认: 0.001)。
  - `--epochs`: 训练轮数 (默认: 1000)。
  - `--alpha`: 控制双重空间损失权重的超参数 (默认: 0.2)。
  - `--tau`: InfoNCE 损失函数中的温度系数 (默认: 0.2)。
  - `--neg`: 对比损失中的负采样数量 (默认: 5)。
# PBFT-CG-MARL 项目部署指南

> 将军你好～这是Mihiro准备的保姆级部署手册，照着一步步来即可 🌸

---

## 📌 重要提醒

- **TPDC 密码**：用环境变量传入，绝不写入任何文件
- **传输方式**：打包成 .tar.gz 整包上传，不在线传文件
- **GPU策略**：无卡模式装环境+下载数据 → 切有卡模式训练
- **数据盘**：50GB 足够（项目代码约 5MB + 训练日志）

---

## 1️⃣ AutoDL 实例开无卡模式

### 步骤1：登录控制台
- 打开 https://www.autodl.com/console
- 手机号 + 密码登录

### 步骤2：租用新实例
| 配置项 | 选择 |
|--------|------|
| 区域 | 就近（延迟低） |
| **GPU** | **"无卡模式"**（省钱！0.1-0.5元/小时） |
| 镜像 | PyTorch 2.1.0 + Python 3.10 + CUDA 12.1 |
| 数据盘 | 50GB → 挂载 `/root/autodl-tmp` |
| 点击 | "创建实例" |

### 步骤3：获取连接信息
- 主机：`region-xxxx.autodl.com`
- 端口：`22xxx`
- 密码：`xxxxxxxx`

---

## 2️⃣ 上传项目代码

### 步骤1：下载Mihiro准备的项目包
- 将军会拿到一个 `PBFT-CG-MARL.tar.gz` 的下载链接

### 步骤2：上传到AutoDL
**方式A：用 JupyterLab（最简单）**
1. AutoDL实例页 → 点击"**JupyterLab**"
2. 左侧文件树 → 进入 `/root/autodl-tmp`
3. 上传 `PBFT-CG-MARL.tar.gz`
4. 打开 Terminal：
```bash
cd /root/autodl-tmp
tar -xzf PBFT-CG-MARL.tar.gz
cd PBFT-CG-MARL
ls -la
```

**方式B：用 SCP（Windows CMD）**
```cmd
cd <你的下载目录>
scp -P <端口> PBFT-CG-MARL.tar.gz root@<主机>:/root/autodl-tmp/
```

### 步骤3：验证解压
```bash
ls /root/autodl-tmp/PBFT-CG-MARL/
```
✅ 应该看到：`configs/  src/  scripts/  README.md  DEPLOY.md  ...`

---

## 3️⃣ 装环境（无卡模式即可）

### 步骤1：创建Conda环境
```bash
cd /root/autodl-tmp/PBFT-CG-MARL
conda create -n pbft python=3.10 -y
conda activate pbft
```
> 💡 命令行前面出现 `(pbft)` = 激活成功

### 步骤2：装PyTorch（CUDA 12.1兼容版）
```bash
# 装PyTorch（清华源加速）
pip install torch==2.1.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤3：装其他依赖
```bash
pip install -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤4：验证安装
```bash
python -c "
import torch
import pettingzoo
import smaclite
import vmas
print('torch:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
print('所有依赖OK！')
"
```
✅ 期望：`torch: 2.1.0` + `CUDA可用: True`（切有卡后）

---

## 4️⃣ 下载数据（双线并行）

### 数据1：GTN-P 地温数据（即拿即用）
```bash
mkdir -p /root/autodl-tmp/gtnp_data
cd /root/autodl-tmp/gtnp_data
git clone https://github.com/.../GTN-P.git  # 或 wget
```

### 数据2：TPDC 冻土数据（需要登录）
```bash
cd /root/autodl-tmp/PBFT-CG-MARL

# ⚠️ 密码走环境变量（不落历史）
export TPDC_EMAIL="fanxiaohu@whcp.edu.cn"
export TPDC_PWD="你的密码"

# 测试登录
python scripts/download_tpdc.py --test-login

# 列出候选数据集
python scripts/download_tpdc.py --list

# 批量下载（需先编辑 urls.txt）
python scripts/download_tpdc.py --download-urls urls.txt
```

---

## 5️⃣ 跑冒烟测试

```bash
cd /root/autodl-tmp/PBFT-CG-MARL
conda activate pbft

# 跑快速测试（30秒）
python scripts/smoke_test.py --with-torch
```

✅ 期望输出：
```
[1/6] 测试算法注册表...  ✅
[2/6] 测试环境注册表...  ✅
[3/6] 测试环境创建...    ✅
[4/6] 测试PBFT共识层...  ✅
[5/6] 测试形状归一化...  ✅
[6/6] 测试完整流程...    ✅
通过: 6/6
🎉 全部通过！可以开始训练
```

---

## 6️⃣ 切到有卡模式（开始训练）

### 步骤1：关机无卡实例
1. AutoDL控制台 → **关机**（数据盘保留）

### 步骤2：开RTX 4090
1. 同镜像重新开机
2. GPU: 1 × RTX 4090

### 步骤3：验证GPU
```bash
nvidia-smi
```
✅ 期望：GPU 0: **RTX 4090, 24GB VRAM**

### 步骤4：端到端小测试（5分钟）
```bash
cd /root/autodl-tmp/PBFT-CG-MARL
conda activate pbft

python src/train.py \
    --algo pbft_cg_mappo \
    --env mpe_spread \
    --n_timesteps 500 \
    --seed 42
```

---

## 7️⃣ 跑全部实验

### 方式A：一键跑全部（推荐）
```bash
cd /root/autodl-tmp/PBFT-CG-MARL
conda activate pbft

# 后台运行（断开SSH也不停）
nohup bash run_all.sh > all_experiments.log 2>&1 &

# 监控进度
tail -f all_experiments.log
# 或
ps aux | grep train.py | grep -v grep
```

### 方式B：单个实验
```bash
python src/train.py \
    --algo pbft_cg_mappo \
    --env smaclite_5m_vs_6m \
    --n_timesteps 1000000 \
    --seed 42
```

---

## 8️⃣ 常见问题FAQ

| 问题 | 解决方案 |
|------|----------|
| `python: command not found` | `conda activate pbft` |
| `No module named 'torch'` | 重装PyTorch（步骤3.2） |
| `CUDA out of memory` | 减小 `hidden_dim` 或 `num_mini_batch` |
| SSH断了训练停 | 用 `tmux` 或 `nohup` 后台运行 |
| 数据下载慢 | 用清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| PBFT报错 `n_agents>=3f+1` | 确保 `n_agents >= 4` |
| 想看训练进度 | `tail -f results/*/progress.csv` |
| 想可视化结果 | `python src/scripts/visualize.py` |

---

## 9️⃣ 文件清单

```
PBFT-CG-MARL/
├── src/                    # 源代码
│   ├── algorithms/         # 6个算法（PBFT-CG-MAPPO核心创新）
│   ├── consensus/          # PBFT共识层（580行核心代码）
│   ├── envs/               # 4个环境适配器（含Fallback）
│   ├── networks/           # Actor/Critic网络
│   ├── utils/              # Buffer/Logger/Metrics
│   ├── scripts/            # 可视化 + GTN-P数据
│   ├── train.py            # 主训练脚本
│   └── eval.py             # 评估脚本
├── configs/                # 配置文件（13个YAML）
├── scripts/                # 辅助脚本
│   ├── smoke_test.py       # 冒烟测试
│   └── download_tpdc.py    # TPDC数据下载
├── run_all.sh              # 一键跑全部实验
├── requirements.txt        # Python依赖
├── DEPLOY.md               # 部署手册
└── README.md               # 项目说明
```

---

## 🔟 时间规划（3天）

| Day | 任务 | 预计耗时 |
|-----|------|----------|
| Day 1 | 部署+环境验证+冒烟测试+1-2个算法跑通 | 6-8小时 |
| Day 2 | 跑完6算法×4环境=24组合+拜占庭实验 | 12-16小时 |
| Day 3 | 消融实验+真实数据应用+论文图表更新 | 8-12小时 |

---

🌟 Mihiro的小贴士：
- 无卡模式费用极低，**先装环境+下载数据**再切有卡
- 训练时用 `tmux` 或 `nohup` 后台跑，断连不中断
- 出问题先看 `DEPLOY.md` 第8节FAQ
- 报告/论文相关问题随时找Mihiro～

将军加油！🚀

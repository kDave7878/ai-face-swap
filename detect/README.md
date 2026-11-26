# 检测模块说明（detect/）

本目录是「基于深度学习的 AI 人脸伪造与对抗防范」课程项目的**检测侧**实现，对应 PPT 实施计划第 03 阶段（训练活体检测 CNN，构建空域+频域双分支检测网络，并评估）。

攻击侧代码在 `../src/`，**未做任何改动**；本模块通过 `importlib` 复用 `src/face_swap_fixed.py` 的换脸流水线来自动生成假样本。

> 下文命令均为 PowerShell，Python 启动器统一用 `py`（若你的环境 `python` 可用则替换即可）。
> 所有命令都在**仓库根目录**执行。

---

## 1. 环境准备

```powershell
# 基础依赖（攻击侧 + 检测侧共用）
py -m pip install numpy opencv-python

# PyTorch（CPU 版即可跑通全流程，PPT 风险应对：云端 GPU 训练、CPU 核心验证）
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 仅 --generator swap（用真实换脸产假样本）与 dlib 68 点对齐需要
py -m pip install dlib-bin
```

再把 `shape_predictor_68_face_landmarks.dat` 放到 `src/` 目录下（下载地址见根目录 README.md）。
**没有这个模型文件时**，人脸定位会自动降级到 OpenCV Haar 级联，`synthetic` 模式仍可完整跑通。

依赖清单见 `requirements.txt`。

---

## 2. 程序清单与用途

### 2.1 可执行入口（命令行直接运行）

| 程序 | 用途 | 产出 |
|---|---|---|
| `build_dataset.py` | 自建 real/fake 人脸数据集：真样本=原始照片人脸裁剪，假样本=换脸/合成产物 | `detect_data/manifest.csv` + `real/` + `fake/` |
| `train.py` | 训练检测器（`liveness` 单分支 / `dual` 空域+频域双分支），按验证集 AUC（同分时 val_loss 更低）存最优权重 | `detect_runs/<tag>/model_best.pt`、`model_last.pt`、`history.csv` |
| `evaluate.py` | 在指定 split 上出指标：ACC / F1 / AUC / EER / APCER / BPCER，并画 ROC、扫阈值 | `detect_runs/eval_<时间戳>/report.csv`、`sweep.csv`、`roc.csv`、`errors.csv` |
| `detect.py` | 对单张图或目录做真伪判定，输出 p_fake、结论和标注图 | `detect_result.csv` + 标注图目录 |

### 2.2 支撑模块（被上面 import，不单独运行）

| 模块 | 职责 |
|---|---|
| `config.py` | 所有公共常量与 CLI 默认值（尺寸、标签约定、切分比例、超参、随机种子） |
| `image_io.py` | 中文路径读写图片（`imdecode`/`imencode` 绕过 `cv2.imread` 的路径限制）、列目录 |
| `face_crop.py` | 人脸定位与对齐：dlib 68 点优先，缺失时 Haar + 双眼仿射兜底，输出 `128×128` 裁剪 |
| `freq_features.py` | 频域特征：分块 `8×8`/`16×16` DCT log 幅度图 + 整幅 FFT log 幅度谱（高通），共 3 通道 |
| `manifest.py` | `manifest.csv` 读写，**不依赖 torch**，方便只做数据准备时快速运行 |
| `models.py` | `LivenessCNN`（严格按 PPT 结构）、`DualBranchNet`（双分支）、权重保存/加载 |
| `dataset.py` | 按**身份组**切分（防身份泄漏导致指标虚高）、类别平衡下采样、增强、`Dataset`/`collate` |
| `inference.py` | 按权重内记录的 `meta` 复现预处理，train / evaluate / detect 三方共用同一套 |
| `metrics.py` | 纯 numpy 实现指标，无 sklearn 依赖：混淆矩阵、AUC（Mann-Whitney 秩）、EER、阈值扫描 |

### 2.3 两个网络结构

- **`liveness`**：PPT 第 10 页原结构 —— `Conv3×3-16 → ReLU → MaxPool → Conv3×3-32 → ReLU → MaxPool → FC128 → ReLU → FC2`，只吃空域图。刻意不加 BN/Dropout，保持可解释，也便于第 04 阶段做 FGSM/PGD 对抗攻击。
- **`dual`**：空域分支 + 频域分支各提特征后拼接，再走 `FC128 → FC2`。用来验证「换脸上采样棋盘格痕迹」这类频域伪影是否带来增益。

---

## 3. 完整跑一遍（推荐顺序）

```powershell
cd c:\Users\30728\Desktop\资料\ai-face-swap-main

# ① 造数据集：把若干张含单人清晰正面的照片放进 gallery\ 目录
py detect\build_dataset.py --gallery gallery --out detect_data --generator swap

# ② 训练双分支（默认 img 64 / freq 128）
py detect\train.py --data detect_data --arch dual --epochs 30

# ③ 评估（自动使用权重里存的验证集最优阈值）
py detect\evaluate.py --ckpt detect_runs\<tag>\model_best.pt --data detect_data --split val

# ④ 推理自己的照片，并导出标注图
py detect\detect.py --ckpt detect_runs\<tag>\model_best.pt 你的照片.jpg --out detect_infer
```

### 没有 dlib / 想先自检流水线

`synthetic` 模式不依赖 dlib，纯程序合成伪影样本，用来验证整条链路：

```powershell
py detect\build_dataset.py --out detect_data --generator synthetic --samples 200
py detect\train.py     --data detect_data --arch dual --epochs 20 --img-size 32 --freq-size 64 --batch 8
py detect\evaluate.py  --ckpt detect_runs\<tag>\model_best.pt --data detect_data --split val
py detect\detect.py    --ckpt detect_runs\<tag>\model_best.pt detect_data --out detect_infer --csv detect_result.csv
```

> 注意：`synthetic` 造的是程序伪影，指标只能证明「代码跑通了」，**不能当作真实检测能力**。

---

## 4. 各程序参数详解

### 4.1 `build_dataset.py`

```
py detect\build_dataset.py [--gallery 目录] [--out 目录] [--generator swap|synthetic]
                           [--crop-size 128] [--samples 120] [--seed ...]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--gallery` | 无 | 原始照片目录，`swap` 模式**必填**。每张图要求能检到一个清晰人脸 |
| `--out` | `detect_data` | 输出目录，内含 `manifest.csv` / `real/` / `fake/` |
| `--generator` | `swap` | `swap`=复用 `src/face_swap_fixed.py` 真实换脸；`synthetic`=程序合成伪影（无需 dlib） |
| `--crop-size` | `128` | 落盘人脸裁剪边长 |
| `--samples` | `120` | `synthetic` 模式的规模参数：身份数 ≈ `samples//3`，每个身份产出 1 真 + 若干假，**总样本数会多于该值** |
| `--seed` | `20261026` | 随机种子 |

`manifest.csv` 列：`path, label, group, generator, source_image`
- `label`：`0`=真实，`1`=伪造
- `group`：**身份组**（底图名），切分时同组不跨集，避免身份泄漏
- `source_image`：假样本提供人脸的来源图，便于评估「对不同来源的泛化」

### 4.2 `train.py`

```
py detect\train.py [--data 目录] [--arch liveness|dual] [--img-size 64] [--freq-size 128]
                   [--epochs 30] [--batch 32] [--lr 1e-3] [--weight-decay 1e-4]
                   [--val-ratio 0.2] [--test-ratio 0.0] [--seed ...]
                   [--device cpu|cuda] [--workers 0] [--no-balance]
                   [--out detect_runs] [--tag 实验名]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--arch` | `dual` | `liveness`=PPT 单分支；`dual`=空域+频域双分支 |
| `--img-size` | `64` | 空域输入边长，CPU 友好 |
| `--freq-size` | `128` | 频域输入边长 |
| `--val-ratio` | `0.2` | 验证集比例（按身份组切） |
| `--test-ratio` | `0.0` | 默认把留出集并入验证集；**要报「独立测试集」指标时设 >0** |
| `--no-balance` | 关 | 关闭类别平衡下采样（默认把多数类降到少数类规模） |
| `--tag` | `arch+时间戳` | 本次实验名，决定 `detect_runs/<tag>/` 目录 |
| `--device` | 自动 | 有 GPU 传 `cuda` |

输出：`model_best.pt`（含 `state_dict` + `model_meta` + `val_threshold` + `train_history`）、`model_last.pt`、`history.csv`。
权重里存了 `meta`，所以 `evaluate.py`/`detect.py` **不需要**再传 `--arch/--img-size/--freq-size`，结构会自动还原，预处理也不会跑偏。

### 4.3 `evaluate.py`

```
py detect\evaluate.py --ckpt 权重 [--data 目录] [--split val|test|all]
                      [--threshold 0.5] [--batch 32] [--device ...] [--out 目录]
```

| 参数 | 说明 |
|---|---|
| `--split` | `val` / `test`（需训练时 `--test-ratio>0`） / `all` |
| `--threshold` | 不填则用权重里存的验证集最优阈值（EER 最小处）；做 ROC 分析时建议保持默认 |

输出文件：
- `report.csv`：整体指标 + 按 `generator` / `source_image` 分组的分解指标
- `sweep.csv`：21 档阈值下的 ACC/F1/EER/APCER/BPCER，用来挑工作点
- `roc.csv`：ROC 曲线点，可直接画图
- `errors.csv`：错分样本清单，**做 PPT 里的错例分析就用这个**

### 4.4 `detect.py`

```
py detect\detect.py --ckpt 权重 <图片或目录 ...> [--threshold ...]
                    [--out 标注图目录] [--csv detect_result.csv] [--batch 32] [--device ...]
```

- `inputs` 位置参数可给多个，**目录会递归收集**图片（`build_dataset` 产出的 `real/ fake/` 结构能直接喂进来）。
- 输入若是**已经裁剪好的人脸图**（检测集本身），Haar/dlib 可能检不到脸，此时自动按整图缩放兜底，结果表 `detector` 列标为 `whole`——这是预期行为，因为训练样本就是裁剪图。
- 标注图文件名带 4 位序号前缀，避免不同子目录同名互相覆盖。

---

## 5. 攻击侧（`src/`，未改动）

供对照，具体见根目录 `README.md`：

| 程序 | 运行方式 |
|---|---|
| `face_landmark_fixed.py` | `py src\face_landmark_fixed.py` — 环境自检（dlib、模型文件、中文路径） |
| `face_swap_fixed.py` | `py src\face_swap_fixed.py <底图> <人脸图> [输出名]` — 68 点对齐 + 仿射 + Alpha 融合换脸 |
| `face_swap_menu.py` | `py src\face_swap_menu.py` — 交互式菜单（1 特征点标注 / 2 换脸） |
| `compare_three_versions.py` | `py src\compare_three_versions.py <底图> <人脸图>` — 三个版本效果对比 |
| `pface_face_fixed.py` | 供上面脚本 import 的算法库，一般不单独运行 |

---

## 6. 常见问题

**Q：`--generator swap` 报「未检测到可用的 dlib 或 shape_predictor_68_face_landmarks.dat」**
装 dlib（Windows 建议 `py -m pip install dlib-bin`）并把 `.dat` 放到 `src/`，这两步缺一个都会走这条提示。急着验证链路可先用 `--generator synthetic`。

**Q：训练时 loss 一直停在 0.693 不动**
样本少时每 epoch 的 optimizer step 极少（32 样本 / batch 16 = 2 step），不是 bug。把 `--batch` 调小到 4~8，或加大 `--epochs`。`train.py` 检测到这种情况会主动提示。

**Q：中文路径下图片读不出来 / dlib 打不开模型**
根目录 README 记录的两大坑。本模块已统一处理：读写走 `image_io.py` 的 `imdecode`/`imencode`，dlib 加载模型前 `os.chdir` 到 `src/` 用相对文件名。

**Q：验证集 ACC 很高但不敢信**
检查是不是身份泄漏：同一张底图的真、假样本必须落在同一 split。本模块的 `group` 列和 `group_split()` 就是为此设计，不要改成按样本随机切分。

**Q：想报独立测试集指标**
训练时加 `--test-ratio 0.2`，再 `evaluate.py --split test`。

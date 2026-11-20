# AI Face Swap · 传统几何换脸基线（含公开工程缺陷的修复）

> 课程实践项目：**《基于深度学习的 AI 人脸伪造与对抗防范》**——攻击侧基线实现
> 用途：学术研究与攻防对抗实验，**不得用于伪造他人身份、制造虚假信息或任何违法用途**

---

## 这个仓库是什么

这是按经典「68 点对齐 + 几何变换 + Alpha 融合」流水线实现的人脸替换基线，**不含 GAN 生成**。

相比常见的开源实现，本仓库修复了一个在公开代码中广泛存在、但少有人定位到根因的缺陷（见下）。

| 文件 | 说明 |
|---|---|
| `src/face_swap_fixed.py` | 换脸核心（命令行版），含坐标缺陷修复与越界保护 |
| `src/face_swap_menu.py` | 交互菜单版（特征点检测 / 换脸），输出带时间戳不覆盖 |
| `src/face_landmark_fixed.py` | 68 点特征点检测（稳健版，检查写盘结果） |
| `src/pface_face_fixed.py` | 视频换脸：抽帧 → 逐帧换脸 → I420 重编码，配 PyQt5 界面 |
| `src/compare_three_versions.py` | 三版效果对照实验，直接输出量化指标 |

---

## 核心贡献：`warpAffine` 对 float64 输入的语义反转

### 现象

换脸合成结果出现大片黑色色块，且修复区出现高亮模糊（smear）。

### 常见但错误的做法

把 `cv2.warpAffine` 的 `borderMode` 从 `BORDER_TRANSPARENT` 改为 `BORDER_REPLICATE`。黑块确实消失，但换成了模糊拖影——**治标**。

### 真实根因

**`cv2.warpAffine` 对 `CV_64F`(float64) 输入时，`WARP_INVERSE_MAP` 的作用方向与 `CV_8U` / `CV_32F` 相反。**

实测（同一 M、同一 border，只改 dtype 与 flag）：

| dtype | flag | 掩模落点 |
|---|---|---|
| uint8 1ch/3ch | INVERSE | ✅ 正确 |
| uint8 | 无 flag | ❌ 错位 |
| **float64** | **INVERSE** | ❌ **错位** |
| float64 | 无 flag | ✅ 正确 |

因为 `get_face_mask()` 返回 **float64**，`warped_mask` 落到了错误位置，掩模面积由 1.37% 错成 9.92%，`combined_mask = max(底图mask, 错位源mask)` 覆盖了「源图范围之外」的大片区域，该区域 warp 采样越界 → 留纯黑。

### 修复

```python
def warp_im(im, M, dshape):
    is_f64 = (im.dtype == np.float64)
    work = im.astype(np.float32) if is_f64 else im      # ★ 关键
    o = np.zeros(dshape, dtype=work.dtype)
    cv2.warpAffine(work, M[:2], (dshape[1], dshape[0]), dst=o,
                   borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
    return o.astype(np.float64) if is_f64 else o
```

再加一道保护：warp 一张全 255 的图得到覆盖掩模，把越界像素从合成中剔除。

### 量化效果（底图 person2 + 源脸 person1）

| 版本 | 融合区 | 引入黑块 |
|---|---|---|
| 原版 float64 + TRANSPARENT | 49,191 px | 23,694 px |
| 改 BORDER_REPLICATE（治标） | 49,191 px | 22 px（但边缘 smear） |
| **掩模转 float32（根治）** | **9,874 px** | **5 px** |

复现：`python src/compare_three_versions.py 底图.jpg 源脸.jpg`

---

## 环境与依赖

```bash
pip install -r requirements.txt
```

需要 `shape_predictor_68_face_landmarks.dat`（dlib 官方模型，约 100 MB，本仓库不包含）：

- 下载：<http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2>
- 解压后放到 `src/` 同目录，**用相对文件名加载**

> ⚠️ 两个已知坑（Windows 实测）：
> 1. `dlib.shape_predictor()` 传**含中文的绝对路径**会报 `RuntimeError: Unable to open`（dlib 用窄字符 fopen）→ 只能 `os.chdir` 到脚本目录后使用相对名。
> 2. 部分 OpenCV 版本下 `cv2.imread()` 对中文路径返回 `None` → 用 `np.fromfile + cv2.imdecode` 代替。

---

## 使用

```bash
# 换脸（命令行）
python src/face_swap_fixed.py 底图.jpg 源脸.jpg [输出名.jpg]

# 交互菜单（特征点检测 / 换脸）
python src/face_swap_menu.py

# 视频换脸（需 PyQt5）
python src/pface_face_fixed.py
```

---

## 量化效果对比

本仓库**不附带任何人脸素材与效果图**（隐私与合规考虑）。运行下列命令即可在本地生成三版对照图与量化指标：

```bash
python src/compare_three_versions.py 底图.jpg 源脸.jpg
```

---

## 免责声明

本项目为**信息安全课程**的攻防对抗研究，目的是理解伪造手段并支撑检测研究。

- 仅使用本人及已获授权的素材进行实验
- **禁止**用于伪造他人身份、诈骗、制造虚假信息、规避人脸识别等任何违法用途
- 使用者须自行承担遵守《网络安全法》《数据安全法》《个人信息保护法》等法规的责任

---

## 致谢

换脸核心算法参考 [matthewearl/faceswap](https://github.com/matthewearl/faceswap)（MIT）；
视频流水线参考 PFace（PyQt5 封装）。本仓库在其基础上修复上述坐标缺陷并做工程化整理。

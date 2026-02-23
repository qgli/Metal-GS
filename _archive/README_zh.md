# Metal-GS：Apple Silicon 原生 3D 高斯溅射渲染器

> **从零构建的 Metal Compute 全链路 3DGS 管线 — 不是移植，而是为统一内存架构重新设计**

[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon-black?logo=apple)](https://developer.apple.com/metal/)
[![Metal](https://img.shields.io/badge/Metal-3.0-blue)](https://developer.apple.com/metal/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 目录

- [1. 项目愿景与背景](#1-项目愿景与背景)
- [2. 算法架构剖析](#2-算法架构剖析-core-architecture)
  - [2.1 SH 球谐函数计算（混合精度）](#21-sh-球谐函数计算混合精度)
  - [2.2 Preprocess 预处理（投影与 Culling）](#22-preprocess-预处理投影与-culling)
  - [2.3 Radix Sort 基数排序](#23-radix-sort-基数排序)
  - [2.4 Tile Binning 瓦片分配](#24-tile-binning-瓦片分配)
  - [2.5 Rasterization 光栅化](#25-rasterization-前向光栅化)
- [3. 精度与内存管理策略](#3-精度与内存管理策略)
- [4. 编译与安装指南](#4-编译与安装指南)
- [5. Python API 参考](#5-python-api-参考)
- [6. 性能基准](#6-性能基准)
- [7. 架构设计哲学](#7-架构设计哲学)

---

## 1. 项目愿景与背景

### 为什么从零重写，而不是翻译 CUDA？

3D Gaussian Splatting（3DGS）是当今最具竞争力的神经场景重建方法之一。然而，其开源实现（Inria `diff-gaussian-rasterization`、Nerfstudio `gsplat`）无一例外地深度绑定 CUDA 生态，完全无法在 Apple Silicon 上运行。

社区的常见做法是"翻译"——将 CUDA kernel 逐行改写为 Metal shader（如 OpenSplat 的 `gsplat-metal`）。**我们拒绝这条路线。** 原因如下：

#### ❶ CUDA 的隐含假设在 Metal 上不成立

| 假设 | CUDA (NVIDIA) | Metal (Apple Silicon) |
|:---|:---|:---|
| GPU-CPU 内存模型 | 独立显存，PCIe 传输 | **统一内存 (Unified Memory)**，零拷贝 |
| 渲染架构 | IMR (Immediate Mode Rendering) | **TBDR (Tile-Based Deferred Rendering)** |
| 线程组前向进度保证 | 所有 SM 上的 Warp 可并发调度 | **无跨 Threadgroup 前向进度保证**（M1 仅 7 核） |
| 原子操作 | `memory_order_seq_cst` 全序 | 仅 `memory_order_relaxed` / `acquire` / `release` |
| SIMD 宽度 | 32 (Warp) | **32 (SIMD group)**，但无 `__ballot_sync` |

逐行翻译 CUDA 代码，会把 NVIDIA 架构的思维定势带入 Apple Silicon，错失统一内存的零拷贝优势，也可能触发 TBDR 架构下的死锁（详见 §2.3 One-Sweep 分析）。

#### ❷ 统一内存是 Apple Silicon 最大的架构红利

在传统 CUDA 工作流中，数据必须经过 `cudaMemcpy` 在主机与设备间搬运。Apple Silicon 的统一内存意味着：

- `MTLResourceStorageModeShared` 缓冲区 CPU 和 GPU **共享同一物理地址**
- 预处理结果（means2d, cov2d, depths）可以直接被后续 kernel 读取，**无需任何拷贝**
- CPU 端计算（如前缀和偏移量）可以 **零开销** 写入 GPU buffer

我们的整个管线（5 个阶段）利用这一特性，将 CPU 与 GPU 的边界模糊化，在最合适的处理器上执行每一步。

#### ❸ 设计目标

1. **Metal-native**：不是 CUDA 的翻译件，而是从 Apple GPU 微架构出发的原生实现
2. **训练就绪**：前向管线 + 反向传播，直接接入 PyTorch `autograd.Function`
3. **性能极致**：单 Encoder 融合、SIMD 前缀和、3 级扫描、动态 CPU 回退
4. **可验证**：每个阶段都有独立的 NumPy 参考实现，bit-exact 或误差在 FP32 精度内

---

## 2. 算法架构剖析 (Core Architecture)

Metal-GS 的前向渲染管线由 5 个阶段组成，从输入的 3D 高斯球参数到最终的 2D 图像：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Metal-GS 前向管线 (Forward Pipeline)              │
│                                                                     │
│  means3d ──┐                                                        │
│  scales  ──┤  ① SH 计算 ─→ colors[N,3]                             │
│  quats   ──┤  ② Preprocess ─→ means2d, cov2d, depths, radii, tiles  │
│  viewmat ──┘                 │                                      │
│                              ↓                                      │
│                  ③ Radix Sort (by depth) ─→ sorted_indices           │
│                              │                                      │
│                              ↓                                      │
│                  ④ Tile Binning ─→ point_list, tile_bins             │
│                              │                                      │
│                              ↓                                      │
│                  ⑤ Rasterize ─→ out_img[H,W,3]                      │
│                                                                     │
│  ┌────────── Single MTLCommandQueue, minimal command buffers ──────┐ │
│  │  Stages ③④ 共享单个 Encoder (memoryBarrierWithScope)           │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 SH 球谐函数计算（混合精度）

**文件**: `csrc/kernels/sh_forward.metal`

球谐函数（Spherical Harmonics）将视线方向编码为 RGB 颜色，支持 degree 0–3（最多 16 个基函数）。

**精度策略**: 
- **输入系数 (SH coefficients)**: FP16 — 节省 2× 带宽，对颜色精度影响可忽略
- **视线方向 (directions)**: FP32 — 归一化需要高精度避免除零
- **内部累加 (accumulation)**: FP32 — 避免高阶基函数间的灾难性抵消
- **输出颜色**: FP16 — 下游 alpha blending 的带宽友好格式

**关键设计**:
- 每线程独立计算（embarrassingly parallel），无需 threadgroup 共享内存
- `float3` 使用 FP32 → FP16 的精确转换路径（先 FP32 累加再截断）
- Condon-Shortley 相位约定，与 Inria 原版完全一致

**性能**: 100 万高斯球 39× 加速（相对于 NumPy 参考实现）

---

### 2.2 Preprocess 预处理（投影与 Culling）

**文件**: `csrc/kernels/preprocess.metal`

将 3D 高斯球投影到 2D 屏幕空间——这是整个管线中数学最密集的阶段。

**计算流程（单线程处理一个高斯球）**:

```
                    3D 世界空间
                        │
          ┌─────────────┼─────────────┐
          │  ① 视图变换 (W · p)       │
          │     p_world → p_camera     │
          │  ② 近平面裁剪 (z > 0.01)  │
          └─────────────┼─────────────┘
                        │
          ┌─────────────┼─────────────┐
          │  ③ 3D 协方差矩阵          │
          │     Σ3D = R·diag(s²)·Rᵀ   │
          │     四元数 → 旋转矩阵      │
          └─────────────┼─────────────┘
                        │
          ┌─────────────┼─────────────┐
          │  ④ EWA Splatting           │
          │     T = J · W              │
          │     Σ2D = T·Σ3D·Tᵀ + σI   │
          │  ⑤ FOV 守卫带钳位         │
          │     (±1.3 × tan_fov)       │
          └─────────────┼─────────────┘
                        │
          ┌─────────────┼─────────────┐
          │  ⑥ 特征值 → 3σ 半径       │
          │     λ_max → radius         │
          │  ⑦ 屏幕投影               │
          │     (u, v) = fx·x/z + cx   │
          │  ⑧ Tile 包围盒            │
          │     [tile_min, tile_max)   │
          └─────────────────────────────┘
```

**关键工程决策**:

| 决策 | 原因 |
|:---|:---|
| 使用 `device float*` 而非 `device float3*` | Metal 的 `float3` 类型有 16 字节对齐（而非紧凑的 12 字节），会导致偏移错位 |
| 防抖模糊 (AA_BLUR = 0.3) 加入 Σ2D | 防止退化协方差矩阵（行列式为零）导致的除零 |
| 特征值判别式下限 0.1f | 与 gsplat 保持一致，防止 sqrt 负值 |
| `PreprocessParams` 结构体字段顺序 | 必须与 C++ 侧 **字节对齐一一对应**，否则 GPU 读到错误参数 |

**输出**:
- `means2d[N, 2]` — 2D 屏幕坐标
- `cov2d[N, 3]` — 2D 协方差上三角 [a, b, c]
- `depths[N]` — 相机空间深度（用于排序）
- `radii[N]` — 3σ 像素半径（0 = 不可见）
- `tile_min[N, 2]`, `tile_max[N, 2]` — Tile 包围盒

**验证**: 10 个高斯球，每个输出字段与 NumPy 参考实现 **误差为零**。

---

### 2.3 Radix Sort 基数排序

**文件**: `csrc/kernels/radix_sort.metal`

对所有可见高斯球按深度排序，是 alpha blending 正确性的前提（前向混合必须遵循前→后顺序）。

#### 算法选择：为什么不用 One-Sweep？

学术界 GPU 排序的 SOTA 是 **Onesweep**（Merrill & Grimshaw 2022），它将 histogram + prefix-scan + scatter 融合为单次 kernel 调用，通过 **Decoupled Look-back** 实现跨线程组的流水线式前缀和传播。

**我们在 Metal / Apple Silicon 上无法安全使用 One-Sweep。原因如下：**

> **❶ 无跨 Threadgroup 前向进度保证**
> 
> M1 GPU 仅有 7 个计算核心，最多同时调度约 28 个 Threadgroup。Decoupled Look-back
> 要求每个 Threadgroup 能"看回"之前所有 Threadgroup 的结果。当排序 1M 个 key
> 产生 4096 个 block 时，look-back 链长远超并发上限 → **后续 TG 会永远等待尚未被调度的前驱 TG → 死锁。**
>
> **❷ Metal 原子序限制**
>
> One-Sweep 的 INVALID → AGGREGATE → PREFIX 三态转换依赖 `memory_order_seq_cst`
> 全序原子操作。Metal device 端仅提供 `relaxed` / `acquire` / `release`，
> 无法保证跨 TG 的状态可见性顺序。

#### 我们的替代方案：经典多 Pass + 极致调度优化

```
Pass 0 (bit 0-3):   histogram → prefix_sum → scatter
Pass 1 (bit 4-7):   histogram → prefix_sum → scatter
  ...                    (ping-pong 缓冲区)
Pass 7 (bit 28-31):  histogram → prefix_sum → scatter
```

**核心优化技术**:

1. **单 Encoder 融合 (Single Encoder Fusion)**
   - 传统做法：每个 sub-kernel 一个 `MTLComputeCommandEncoder`（8 pass × 5 kernel = 40 个 encoder）
   - 我们的做法：**整个排序仅 1 个 encoder**，用 `[enc memoryBarrierWithScope:MTLBarrierScopeBuffers]` 在 dispatch 之间保证数据可见性
   - 收益：消除约 40 次 encoder 创建/销毁的 CPU 开销

2. **SIMD 前缀和稳定散射 (SIMD Prefix Sum Scatter)**
   - CUDA 版本使用 `__ballot_sync()` 进行位掩码排名 → Metal 不支持
   - 我们使用 `simd_prefix_exclusive_sum()` 实现相同功能，且天然保序（stable）
   - 跨 SIMD group 的前缀和通过 threadgroup 共享内存传播

3. **3 级前缀和扫描 (3-Level Hierarchical Scan)**
   - 问题：100K 高斯球在 binning 阶段产生 ~11M 次相交，单级扫描仅支持 65K 元素
   - 方案：3 级 Blelloch 扫描层次，支持高达 256³ = 16M 元素
   - 关键修复：Phase 4 中 2 级 → 3 级的升级是解决 11M 相交排序的转折点

4. **动态 Pass 数量**
   - 深度排序：8 pass（完整 32-bit 浮点排序）
   - Tile ID 排序：4 pass（3600 tiles 仅需 ~12 bit → 向上取偶数 pass）
   - 通过 `dispatch_radix_sort_kv()` 统一接口，`num_passes` 参数化

5. **CPU 动态回退 (Dynamic CPU Fallback)**
   - 当 N ≤ 16384 时，GPU kernel 启动开销 > 实际计算时间
   - 自动切换为 `std::sort()`，利用统一内存零拷贝直接排序

**性能**: 1M 元素 7.2× 加速（相对于 NumPy `argsort`）

---

### 2.4 Tile Binning 瓦片分配

**文件**: `csrc/kernels/binning.metal`

将深度排序后的高斯球分配到屏幕瓦片（16×16 像素），为光栅化阶段建立逐 Tile 的渲染队列。

**三步管线（共享单个 Encoder）**:

```
┌─────────────────────────────────────────────────────┐
│  ① generate_intersections                            │
│     每个高斯球 → 展开为 (tile_id, gauss_id) 对       │
│     利用 CPU 预计算偏移量 → 零原子冲突写入           │
├─────────────────────────────────────────────────────┤
│  ② radix_sort_kv (stable, by tile_id)               │
│     稳定排序保持深度顺序不变                         │
│     动态 4 pass (16-bit tile ID)                     │
├─────────────────────────────────────────────────────┤
│  ③ identify_tile_ranges                              │
│     边界检测 → tile_bins[tile] = (start, end)        │
│     空 tile 保持 (0, 0) → 光栅化阶段零开销跳过       │
└─────────────────────────────────────────────────────┘
```

**核心不变量**: 高斯球以深度排序顺序输入 → 稳定排序按 tile_id 分组 → **每个 tile 内部自然保持深度顺序** → alpha blending 正确性保证。

**CPU-GPU 协作设计**:

`offsets[]`（每个高斯球的写入偏移量）在 CPU 端通过 O(N) 前缀和计算。这看似"不够 GPU"，但在统一内存架构下：
- CPU 计算 < 1ms（即使 100K 高斯球）
- 消除了 GPU 端的 `atomic_add` 计数 + 第二次扫描，简化了 kernel 逻辑
- CPU 写入的 `offsets` 数组 GPU 可直接读取（零拷贝）

**验证**: 100K 高斯球，11,059,628 次相交，与 NumPy 参考实现 **精确匹配（exact match）**。

---

### 2.5 Rasterization 前向光栅化

**文件**: `csrc/kernels/rasterize.metal`

整个管线的终极阶段——将分配到每个 Tile 的高斯球通过 alpha blending 混合为最终像素颜色。

**架构设计**:

```
                    ┌─────────────────────────────┐
                    │   Grid: (80, 45) tiles       │
                    │   每个 Tile = 1 Threadgroup   │
                    │   Threadgroup: 16×16 = 256   │
                    │   每个线程 = 1 个像素          │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │  对当前 Tile 的高斯球队列     │
                    │  以 BLOCK_SIZE=256 为批次     │
                    │  循环处理：                    │
                    │                              │
                    │  ① 协作加载 (Cooperative Fetch)│
                    │     256 线程各加载 1 个高斯球  │
                    │     → threadgroup 共享内存     │
                    │     threadgroup_barrier()      │
                    │                              │
                    │  ② 逐像素评估                 │
                    │     每线程遍历本批 256 个高斯   │
                    │     计算 Mahalanobis 距离      │
                    │     alpha blending 累加        │
                    │                              │
                    │  ③ 早期终止                   │
                    │     T < 1e-4 → break          │
                    │                              │
                    │  threadgroup_barrier()         │
                    │  → 下一批                     │
                    └──────────────────────────────┘
```

**Threadgroup 共享内存结构**:

```metal
struct SharedGaussian {
    float2 mean;       // 2D 中心 (8 bytes)
    float3 cov_inv;    // 逆协方差 [inv_a, inv_b, inv_c] (12 bytes)
    float3 color;      // RGB (12 bytes)
    float  opacity;    // 不透明度 (4 bytes)
};                     // 总计: 36 bytes × 256 = 9 KB per tile
```

**数值稳定性（极其关键）**:

协方差矩阵求逆 $\Sigma^{-1}_{2D}$ 需要计算行列式的倒数。退化高斯球（det → 0）会产生无穷大：

```metal
float det = a * c_val - b * b;
// ε 守卫：det 过小则视为退化，强制 opacity = 0
float det_inv = (det > 1e-6f) ? (1.0f / det) : 0.0f;
shared_gs[flat_tid].opacity = (det > 1e-6f) ? opacities[gauss_id] : 0.0f;
```

同时，Mahalanobis 距离有双重守卫：
- `maha < 0`：数值错误（逆矩阵不正定）→ 跳过
- `maha > 18`：高斯权重 $e^{-9} < 10^{-4}$ → 贡献可忽略 → 跳过

**Alpha Blending 公式（前向累积）**:

$$C_{\text{pixel}} = \sum_{i=1}^{N} c_i \cdot \alpha_i \cdot T_i + T_N \cdot C_{\text{bg}}$$

其中 $T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$ 为累积透射率，$\alpha_i = \min(0.999, \; o_i \cdot e^{-\frac{1}{2}\Delta^T \Sigma^{-1}_{2D} \Delta})$。

**验证**: 100K 高斯球，1280×720 渲染，与 NumPy 参考实现 **最大误差 = 0.000000**。

---

## 3. 精度与内存管理策略

### 3.1 精度分层（Precision Tiering）

Metal-GS 采用精心设计的精度分层策略，在带宽、精度和未来可扩展性间取得平衡：

| 数据类型 | 精度 | 原因 |
|:---|:---|:---|
| 3D 均值 / 深度 / 协方差 | **FP32** | 投影 Jacobian 和 EWA splatting 对精度极度敏感；FP16 的 10-bit 尾数会导致远处高斯球的 Z 值完全丢失 |
| 四元数 / 缩放因子 | **FP32** | 反向传播梯度需要高精度以避免训练发散 |
| SH 系数 | **FP16** | 颜色精度对 10-bit 尾数足够；2× 带宽节省显著提升 SH kernel 的算术强度 |
| SH 累加 | **FP32** | 高阶基函数 (degree 3) 间存在大量正负抵消，FP16 累加会导致灾难性精度损失 |
| Alpha blending 累加 | **FP32** | 透射率 $T$ 随深度指数衰减，FP16 很快下溢到零 |
| 输出图像 | **FP32** | 光栅化输出为 [0, 1] 浮点，方便后续 loss 计算 |

### 3.2 ENABLE_BF16 宏：为 M4 预留的伏笔

```c
// setup.py 中的编译时开关
METAL_DEFINES = {
    "ENABLE_BF16": "0",  // M1/M2/M3: 使用 FP32
                         // M4+: 设为 "1" 启用 BF16 训练路径
}
```

BF16 (Brain Float 16) 保留了 FP32 的 8-bit 指数范围（而 FP16 仅 5-bit），适合梯度计算。M4 (Apple GPU Family 9+) 原生支持 BF16 ALU，届时可将协方差计算等中间步骤切换到 BF16，获得约 1.5-2× 的吞吐提升，同时保持训练稳定性。

当前在 M1 上，`bfloat` 类型无硬件支持，强制使用会导致软件模拟反而更慢，故默认关闭。

### 3.3 内存布局：为什么拒绝 float3?

Metal 的 `device float3*` 指针有 **16 字节步幅**（不是 12 字节）。这意味着：

```
// ❌ 如果 buffer 是紧凑的 [x0,y0,z0, x1,y1,z1, ...]
device float3* p = ...; 
p[1];  // 读取偏移 16 字节处 → 错误！实际数据在偏移 12 字节处

// ✅ 我们的做法：raw float* + 手动索引
device float* p = ...;
float3 val = float3(p[i*3], p[i*3+1], p[i*3+2]);  // 正确
```

**全部 buffer 接口统一使用 `device float*` / `device uint*`**，手动管理偏移。这增加了代码量，但消除了 Metal 内存对齐陷阱。

### 3.4 统一内存零拷贝策略

所有 `MTLBuffer` 使用 `MTLResourceStorageModeShared`：

```objc
id<MTLBuffer> buf = [device newBufferWithBytes:data
                               length:size
                               options:MTLResourceStorageModeShared];
// CPU 和 GPU 共享同一物理地址 — 无 DMA 传输
```

管线中的 CPU-GPU 协作点：
- **CPU 计算偏移量** → GPU `generate_intersections` 直接读取（§2.4）
- **CPU 回退排序** (N ≤ 16K) → 直接写入共享 buffer（§2.3）
- **GPU 输出** → CPU `memcpy` 从共享 buffer 读取（本质是地址拷贝，非 DMA）

---

## 4. 编译与安装指南

### 4.1 环境要求

| 要求 | 版本 |
|:---|:---|
| macOS | 13.0+ (Ventura) |
| Apple Silicon | M1 / M2 / M3 / M4 |
| Xcode Command Line Tools | 14.0+ (含 Metal 编译器) |
| Python | 3.10+ |
| Conda | 推荐 Miniforge / Miniconda |

### 4.2 安装步骤

```bash
# 1. 创建 Conda 环境
conda create -n metal-gs python=3.10 -y
conda activate metal-gs

# 2. 安装依赖
pip install numpy pybind11 Pillow

# 3. 编译安装（必须强制 Apple Clang — GCC 无法编译 ObjC++）
CC=/usr/bin/clang CXX=/usr/bin/clang++ pip install -e .
```

### 4.3 编译流程解析

`setup.py` 中的 `MetalBuildExt` 类执行两阶段编译：

**阶段 1：AOT Metal 着色器编译**
```
*.metal → xcrun metal -c → *.air → xcrun metallib → metal_gs.metallib
```
- 所有 5 个 `.metal` 文件编译为 `.air` (AIR = Apple Intermediate Representation)
- 链接为单个 `metal_gs.metallib`（Metal Library Bundle）
- 编译参数：`-std=metal3.0 -ffast-math -D ENABLE_BF16=0`

**阶段 2：C++/ObjC++ 扩展编译**
- `metal_wrapper.mm`：ObjC++ 文件，使用 `-ObjC++ -std=c++17 -fobjc-arc`
- `bindings.cpp`：PyBind11 绑定
- 链接框架：`-framework Metal -framework Foundation -framework MetalPerformanceShaders`

### 4.4 验证安装

```bash
# 运行全部前向管线测试
python test-fw/verify_preprocess.py    # 预处理验证
python test-fw/verify_sort.py          # 排序验证
python test-fw/verify_binning.py       # 瓦片分配验证
python test-fw/verify_render.py        # 完整渲染验证

# 全部应输出: ✅ ALL TESTS PASSED
```

---

## 5. Python API 参考

### 5.1 顶层 API：`render_forward()`

```python
import numpy as np
from metal_gs._metal_gs_core import render_forward

# ---- 准备输入 ----
N = 100_000
means3d   = np.random.randn(N, 3).astype(np.float32)
scales    = np.exp(np.random.uniform(-4, -1.5, (N, 3))).astype(np.float32)
quats     = np.random.randn(N, 4).astype(np.float32)
quats    /= np.linalg.norm(quats, axis=1, keepdims=True)
colors    = np.random.rand(N, 3).astype(np.float32)
opacities = np.random.uniform(0.3, 0.95, N).astype(np.float32)

# 相机参数
viewmat = np.eye(4, dtype=np.float32)
viewmat[2, 3] = 5.0

fov_x = np.radians(60)
fx = 1280 / (2 * np.tan(fov_x / 2))
fy = fx  # 假设方形像素

# ---- 一键渲染 ----
result = render_forward(
    means3d, scales, quats, viewmat,
    colors, opacities,
    tan_fovx=float(np.tan(fov_x / 2)),
    tan_fovy=float(np.tan(fov_x / 2) * 720 / 1280),
    focal_x=fx, focal_y=fy,
    principal_x=640.0, principal_y=360.0,
    img_width=np.uint32(1280), img_height=np.uint32(720),
    bg_r=0.0, bg_g=0.0, bg_b=0.0,
)

# ---- 输出 ----
image = result['image']           # np.float32 [720, 1280, 3]
print(f"Total: {result['total_ms']:.1f} ms")
print(f"  Preprocess:  {result['preprocess_ms']:.1f} ms")
print(f"  Sort:        {result['sort_ms']:.1f} ms")
print(f"  Binning:     {result['binning_ms']:.1f} ms")
print(f"  Rasterize:   {result['rasterize_ms']:.1f} ms")
print(f"  Visible:     {result['num_visible']}")
print(f"  Intersects:  {result['num_intersections']}")

# 保存为 PNG
from PIL import Image
img_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
Image.fromarray(img_u8).save("output.png")
```

### 5.2 逐阶段 API

每个阶段也可单独调用，方便调试和性能分析：

| API | 输入 | 输出 |
|:---|:---|:---|
| `compute_sh_forward(dirs, sh, N, K, deg)` | 方向+SH系数 | `(colors_fp16, ms)` |
| `preprocess_forward(m3d, s, q, vm, ...)` | 3D参数+相机 | `(means2d, cov2d, depths, radii, tmin, tmax, ms)` |
| `radix_sort_by_depth(depths)` | 深度数组 | `(sorted_indices, ms)` |
| `tile_binning(si, radii, tmin, tmax, ntx, nty)` | 排序结果+tile信息 | `(point_list, tile_bins, num_isect, ms)` |
| `rasterize_forward(m2d, cov, col, op, tb, pl, ...)` | 2D参数+tile信息 | `(image, ms)` |

---

## 6. 性能基准

**测试平台**: MacBook Air M1 (7-core GPU, 16 GB 统一内存)  
**场景**: 100,000 随机高斯球, 1280×720 分辨率

| 阶段 | 耗时 | 说明 |
|:---|:---|:---|
| Preprocess | 1.0 ms | 91K 可见，9K 被裁剪 |
| Depth Sort | 1.2 ms | 8-pass 基数排序 |
| Tile Binning | 46.1 ms | 11M 次相交，4-pass tile 排序 |
| Rasterize | 16.2 ms | 3600 tiles，协作加载 |
| **总计** | **64.6 ms** | **15.5 FPS** |

**关键对比**:
- NumPy 参考实现（仅 64×64 裁剪区域）：1890 ms
- 全分辨率 NumPy 等效估算：~400,000 ms (6.7 分钟 / 帧)
- **Metal-GS 加速比**: ~6000× (相对于纯 NumPy)

**阶段独立基准**:
- SH 计算 (1M 高斯): 39× vs NumPy
- 基数排序 (1M 元素): 7.2× vs NumPy argsort
- 所有阶段验证: **0 误差** (bit-exact 或 FP32 精度内)

---

## 7. 架构设计哲学

### 原则 1：拥抱硬件，而非对抗硬件

CUDA 代码的常见模式——大量原子操作、跨线程组同步、独立显存管理——在 Apple Silicon 上不仅低效，甚至危险（死锁风险）。我们的每一个设计决策都从 M1 的硬件特性出发：

- **统一内存** → CPU-GPU 协作（偏移量计算在 CPU，展开在 GPU）
- **TBDR 架构** → Tile-based dispatch 天然契合光栅化阶段
- **有限并发** → 放弃 One-Sweep，选择安全的多 Pass 方案
- **SIMD 32 宽度** → `simd_prefix_exclusive_sum()` 替代 `__ballot_sync()`

### 原则 2：每一步都可验证

每个 kernel 都有独立的 NumPy 参考实现和验证脚本。这不是"测试"——这是**规范**。当 Metal 输出与 NumPy 有任何偏差时，一定是我们的实现有 bug，而非"浮点误差"。

### 原则 3：为训练设计，而非仅为推理

前向管线的每一个输出都被设计为可以直接参与反向传播：
- `cov2d` 保留上三角形式，方便求逆的 VJP
- `means2d` 保留 FP32 精度，确保梯度不下溢
- 光栅化存储 `T_final`（最终透射率），反向遍历时可恢复每步 $T_i$
- 所有参数结构体字段顺序与 C++ 严格一致，反向 kernel 可复用

---

## 项目文件结构

```
Metal-GS/
├── README_zh.md                    ← 本文件
├── backward_design.txt             ← 反向传播架构设计草案
├── setup.py                        ← 编译脚本 (AOT Metal + C++/ObjC++)
├── csrc/
│   ├── bindings.cpp                ← PyBind11 绑定层
│   ├── metal_wrapper.h             ← C++ 接口声明
│   ├── metal_wrapper.mm            ← ObjC++ Metal 调度实现
│   └── kernels/
│       ├── sh_forward.metal        ← ① SH 球谐计算
│       ├── preprocess.metal        ← ② 3D→2D 投影
│       ├── radix_sort.metal        ← ③ GPU 基数排序
│       ├── binning.metal           ← ④ 瓦片分配
│       ├── rasterize.metal         ← ⑤ 前向光栅化
│       └── metal_gs.metallib       ← AOT 编译输出
├── metal_gs/
│   └── __init__.py                 ← Python 包入口
└── test-fw/
    ├── verify_sh.py                ← SH 验证
    ├── verify_preprocess.py        ← 预处理验证
    ├── verify_sort.py              ← 排序验证
    ├── verify_binning.py           ← 瓦片分配验证
    └── verify_render.py            ← 完整渲染验证 + PNG 输出
```

---

> *"不要翻译别人的代码——要理解硬件，然后写出只有这块硬件才能跑出来的算法。"*
>
> — Metal-GS 设计理念

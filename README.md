# TinyProfiler-GEMM

> 一个单文件、零第三方依赖（可选 OpenMP）的 GEMM（矩阵乘法）体系结构优化实验：
> 从朴素 `i-j-k` 三重循环出发，逐步加入**循环重排**、**Cache Line 对齐**、
> **向量化 / FMA 融合**、**矩阵分块（Cache Blocking）**、**AVX2 寄存器分块微内核**、
> **Packing 数据重排** 与 **OpenMP 多线程**，把 `8192×8192` 大矩阵的吞吐从
> 断崖式下跌的 **6.4 GFLOPS** 一路推到 **140 GFLOPS**（约 **22×**），
> 其中单线程 Packing 已达 **39 GFLOPS**（约 6×）。

---

## 目录

1. [项目简介](#1-项目简介)
2. [核心结果](#2-核心结果)
3. [硬件与环境](#3-硬件与环境)
4. [环境依赖](#4-环境依赖)
5. [构建与运行](#5-构建与运行)
6. [实现演进](#6-实现演进)
7. [性能对比](#7-性能对比)
8. [体系结构优化原理](#8-体系结构优化原理)
9. [正确性校验](#9-正确性校验)
10. [WSL2 的 PMU 限制](#10-wsl2-的-pmu-限制)
11. [后续优化方向](#11-后续优化方向)
12. [文件结构](#12-文件结构)

---

## 1. 项目简介

本项目实现了一个单文件 GEMM baseline 及其逐级优化：

- **计算**：`C = A × B`，`A/B/C` 均为 `N×N` 的 `double` 方阵，`N` 通过命令行传入（默认 `1024`）；
- **存储**：行主序（row-major），`A[i][k] = A[i * N + k]`；
- **计时**：`std::chrono::steady_clock`，毫秒级（含小数）；
- **指标**：`GFLOPS = 2·N³ / (t_ms·1e-3) / 1e9`，其中 `2·N³` 为乘加各算一次的浮点运算总数；
- **校验**：各优化阶段与 `i-k-j` 结果做**全矩阵相对误差**比对，另与独立点积做抽样比对。

优化策略按 Phase 递进，每个 Phase 保持数值结果一致，只改变访存模式、数据布局与并行方式。

---

## 2. 核心结果

在 Intel Core Ultra 7 255H 上（GCC 15.2，`-O3 -march=native -ffast-math`）：

| 阶段 | 实现 | 1024² | 4096² | 8192² |
|------|------|------:|------:|------:|
| Phase 1 | `i-j-k` 朴素 | 0.99 GFLOPS | — | — |
| Phase 3 | `i-k-j` + 对齐 + AVX2/FMA | 14.4 | 6.6 | **6.4** |
| Phase 4 | + 矩阵分块 (block=64) | 20.5 | 16.2 | 13.4 |
| Phase 5 | + AVX2 微内核 (8×4) | 31.4 | 21.2 | 16.9 |
| Phase 6 | + Packing 微内核 (4×8) | **54.1** | **52.7** | **39.1** |
| Phase 7 | + OpenMP 多线程 (16 线程) | 201.8 | 309.9 | **140.3** |

> 关键观察：
> 1. 未分块的 `i-k-j` 在矩阵增大到 `8192²` 时 GFLOPS 从 14.4 跌到 6.4（容量/冲突 Miss + TLB 压力）；
> 2. **矩阵分块 + Packing** 后，单线程 8192² 仍保持 **39.1 GFLOPS**（1024² 为 54.1），断崖基本消除；
> 3. **OpenMP 16 线程**把 8192² 再推到 **140.3 GFLOPS**，相对原始 `i-k-j` 加速约 **22×**。
> 数据为同一机器单次运行值，受睿频 / 温度影响有 ±5% 左右波动。

---

## 3. 硬件与环境

| 项目 | 配置 |
|------|------|
| CPU | Intel Core Ultra 7 255H（Arrow Lake） |
| 主机 OS | Windows 11 Home（24H2） |
| 编译器 | GCC 15.2.0（MinGW-W64 x86_64-ucrt-posix-seh） |
| 构建系统 | CMake ≥ 3.16 |
| 语言标准 | C++17（`-std=c++17`，禁用 GNU 扩展） |
| 优化标志 | `-O3 -march=native -ffast-math`（MSVC：`/O2 /arch:AVX2 /fp:fast`） |
| 向量化 | AVX2 + FMA（Phase 5：`8×4`；Phase 6/7：`4×8`） |
| 并行 | OpenMP（默认开启，最大 16 线程） |

---

## 4. 环境依赖

- **CMake** ≥ 3.16
- **GCC / Clang**（支持 C++17 与 `-march=native`；MSVC 亦有等价分支）
- **OpenMP**（可选，Phase 7 用；缺失时自动退化为单线程）
- （可选）**Linux `perf`** —— 硬件计数器分析，见 [第 10 节](#10-wsl2-的-pmu-限制)

---

## 5. 构建与运行

### Release（推荐）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 默认 N=1024, block=64
./build/gemm

# 指定矩阵维度与分块大小
./build/gemm 8192 64

# 自动扫描 block 大小 (Phase 6)
./build/gemm 4096 64 --tune

# 关闭 OpenMP (纯单线程)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGEMM_OPENMP=OFF
cmake --build build
```

### 命令行参数

| 参数 | 含义 | 默认 |
|------|------|------|
| `argv[1]` | 方阵维度 `N` | 1024 |
| `argv[2]` | 分块大小 `block` | 64 |
| `--tune` | 扫描 `{32,48,64,96,128,160,192,256}` 并给出建议 | 关闭 |

> `-march=native` 会针对当前 CPU 生成指令，换机器可能触发 `Illegal instruction`；
> 可移植产物请改用 `-march=x86-64-v3`（AVX2）或 `-march=x86-64-v4`（AVX-512）。
> 无 AVX2/FMA 的机器上自动退化为标量分块实现。

---

## 6. 实现演进

### Phase 1 — Baseline（`i-j-k`）

```cpp
for (i) for (j) for (k)
    C[i*N+j] += A[i*N+k] * B[k*N+j];
```

最内层 `k` 循环访问 `B[k*N+j]`：相邻两次访问间隔 `N × 8B`，彻底丧失空间局部性；
当 `N` 为 2 的幂（如 1024）时，8 KB 行跨距还会命中 Cache Set 冲突。

### Phase 2/3 — 循环重排（`i-k-j`）+ 对齐 & SIMD

```cpp
for (i) for (k) {
    const double a = A[i*N+k];
    for (j) C[i*N+j] += a * B[k*N+j];
}
```

- 最内层 `j` 循环对 `B` 实现 Stride-1 顺序访存；
- **64B 对齐**：`malloc` 多分配，原始指针存于对齐地址之前，`free` 释放原始指针；
- **AVX2/FMA**：`-march=native` 激活 AVX2/FMA，`a*b+c` 融合为 `vfmadd231pd`，配合 `-ffast-math`。

### Phase 4 — 矩阵分块（Cache Blocking / Tiling）

```cpp
for (i0) for (j0) for (k0)
    for (i in tile) for (k in tile) for (j in tile)
        C[i][j] += A[i][k] * B[k][j];
```

`i0 → j0 → k0` 顺序保证 `C` 的 `block×block` tile 在整个 `k0` 循环期间常驻 cache，
`B` 的重复 DRAM 读取从 `N³` 降到 `N³/block`，消除容量 Miss 与 TLB 压力。

### Phase 5 — AVX2 寄存器分块微内核

```cpp
// MR=8 行 × NR=4 列
for (k) {
    bv = load(B[k][j..j+4]);
    for (r in 0..MR)
        acc[r] = fmadd(broadcast(A[i+r][k]), bv, acc[r]);
}
```

`MR×NR` 个累加器常驻 YMM 寄存器，`C` 只在进出微内核时 load/store 一次。

### Phase 6 — Packing 数据重排 + 微内核（4×8）

```cpp
// A 面板按列优先打包: A_pack[k*MC + r] = A[i0+r][k0+k]
// B 面板按行打包:     B_pack[k*NC + c] = B[k0+k][j0+c]
for (k) {
    b0 = load(B_pack[k][j..j+4]); b1 = load(B_pack[k][j+4..j+8]);
    for (r in 0..MR)
        acc[r][0..1] = fmadd(broadcast(A_pack[k][i+r]), b0/b1, acc[r]);
}
```

把 `A` 面板按列优先重排，使固定 `k` 的 4 个 `A` 元素连续落在同一条 Cache Line 上
（广播只碰 1 条 Line）；`B` 面板整行 `memcpy` 搬入。列宽从 4 扩到 8（两个 YMM），
每个 `A` 广播被 2 个 FMA 共享，广播:FMA 从 1:1 降到 1:2，单线程吞吐大幅提升。

### Phase 7 — OpenMP 多线程

```cpp
#pragma omp parallel for schedule(static)
for (i0 = 0; i0 < N; i0 += block)
    packed_i0_block(i0);   // 每个线程处理若干 i0 行块, 写互不重叠的 C 行
```

按 `i0` 行块并行，各线程写互不重叠的 `C` 行，无需锁。默认 16 线程。

---

## 7. 性能对比

| 阶段 | 循环/布局 | 优化手段 | 1024² | 4096² | 8192² |
|------|----------|---------|------:|------:|------:|
| Phase 1 | `i-j-k` | 无 | 0.99 | — | — |
| Phase 3 | `i-k-j` | 对齐 + AVX2 + FMA | 14.4 | 6.6 | 6.4 |
| Phase 4 | `i-k-j` 分块 | Cache Blocking(64) | 20.5 | 16.2 | 13.4 |
| Phase 5 | 微内核 8×4 | AVX2 寄存器分块 | 31.4 | 21.2 | 16.9 |
| Phase 6 | 微内核 4×8 | + Packing 数据重排 | 54.1 | 52.7 | 39.1 |
| Phase 7 | + OpenMP | 16 线程并行 i0 | 201.8 | 309.9 | 140.3 |

8192² 加速链：`Phase3 → Phase6 ≈ 6.1×`，`Phase3 → Phase7 ≈ 21.9×`。

**加速来源分解**：

- 循环重排：把 `B` 的跨行随机访存变为 Stride-1（约 8~15×）；
- Cache Blocking：`B` 重复 DRAM 读取从 `N³` 降到 `N³/block`，消除容量 Miss 与 TLB 压力；
- 寄存器分块 + Packing：`C` 重复 load/store 从 `N³` 降到 `N²`，`A` 广播命中同一条 Cache Line，
  FMA 指令占比最大化，单线程逼近 AVX2 峰值；
- OpenMP：按行块并行，多核带宽叠加。

---

## 8. 体系结构优化原理

### 8.1 空间局部性与 Cache Line

连续访问 `B[k*N+j]` 时 8 个 `double`（64B）共享一条 Cache Line，空间局部性 100% 被利用；
`i-j-k` 内层跨行访问一次只用一个 Cache Line，浪费带宽。

### 8.2 64 字节对齐

消除跨 Cache Line 的首行访问分裂，允许发射对齐向量加载 `vmovapd`。

### 8.3 AVX2 向量化与 FMA

- **AVX2**：256-bit 寄存器一次处理 4 个 `double`；
- **FMA**：`乘 + 加` 融合为一条指令，只做一次舍入；
- Phase 6 把列宽扩到 8（两个 YMM），广播:FMA 降到 1:2，指令发射更平衡。

### 8.4 Cache Blocking 如何消除 8192² 断崖

`8192×8192` 的 `B` 约 512 MB，远超 L2/L3。未分块 `i-k-j` 对每个 `i` 都把整个 `B`
重读一遍（约 `N³×8B ≈ 4 TB`），且每行 64 KB 跨 16 个 4 KB 页，TLB Miss 爆炸——
这就是 GFLOPS 断崖的来源。分块后 `B` 以 32 KB tile 在 `i` 块间复用，`B` 流量降到
`N³/block×8B ≈ 69 GB`；Packing 再让 `A`/`B` 访问完全连续，单线程 8192² 保持 39 GFLOPS。

### 8.5 `-ffast-math` 的作用

严格 IEEE-754 禁止重排/融合，会阻断自动向量化与 FMA。`-ffast-math` 放宽约束解锁两者，
代价是结果与严格顺序相差数个 ULP——校验因此改用**相对误差**而非精确 `==`。

---

## 9. 正确性校验

每次运行做两层校验：

1. **全矩阵比对**：各优化阶段与 `C_ikj` 逐元素比较最大相对误差；
2. **独立抽样**：随机取 16 个元素与独立点积 `Σ A[i][k]·B[k][j]` 比对。

示例（`N=8192`）：

```
[Phase 6] + Packing 微内核 (4x8)     :  28098.6 ms  (39.130 GFLOPS)
    全矩阵校验 vs i-k-j: max 相对误差 = 0.000e+00  OK
[Phase 7] + OpenMP 多线程            :   7838.5 ms  (140.271 GFLOPS)
    全矩阵校验 vs i-k-j: max 相对误差 = 0.000e+00  OK
抽样校验 (16 个随机元素 vs 独立点积): max 相对误差 = 1.624e-14  OK
```

各优化实现与 `i-k-j` 对每个 `C[i][j]` 都按 `k` 升序累加、同样使用 FMA，故结果逐位一致；
与独立点积的差异在 `1e-14` 量级，属正常浮点噪声。断言阈值 `1e-9` 远大于噪声、远小于可观测误差。

---

## 10. WSL2 的 PMU 限制

在 WSL2 中尝试抓取硬件计数器：

```bash
perf stat -e L1-dcache-load-misses,instructions,cycles ./build/gemm
```

报错 `Error: No supported events found.`。原因：WSL2 运行在 Hyper-V 之上，宿主 Hypervisor
独占 PMU，默认不透传给客户机；`perf list` 仅有 software/tracepoint/kprobe/msr 等软件事件，
`/sys/bus/event_source/devices/` 缺少 `cpu` PMU 设备。这不是 `perf_event_paranoid` 权限问题。

**替代路径**：Windows 原生 + Intel VTune、原生 Linux、或 `-cpu host` 直通 PMU 的 KVM 虚拟机。

---

## 11. 后续优化方向

8192² 已达 140 GFLOPS，但距该 CPU 理论峰值仍有空间：

1. **规范 Packing 顺序（消除冗余 B 打包）**：当前为便于并行，`B` 面板在每个 `i0` 块重复打包
   （总 `O(N³/block)`）；改为 `k0 → j0 → i0` 的 GotoBLAS 顺序可把 `B` 打包降到 `O(N²)`，进一步降低内存流量；
2. **AVX-512 微内核**：若 CPU 支持，列宽可再翻倍；
3. **NUMA / 亲和性与频率调优**：绑定物理核、关闭 E-core 干扰，缓解笔记本功耗/散热对并行的拖累；
4. **Roofline 定位**：实测内存带宽与峰值算力，判断 8192² 的瓶颈在访存还是计算。

---

## 12. 文件结构

```
TinyProfiler-GEMM/
├── CMakeLists.txt   # 构建脚本 (-O3 -march=native -ffast-math, 可选 OpenMP)
├── main.cpp         # GEMM 单文件实现 (7 个 Phase + 计时 + 双层校验)
├── README.md        # 本实验报告
└── .gitignore       # 忽略构建产物
```

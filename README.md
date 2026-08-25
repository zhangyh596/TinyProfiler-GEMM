# TinyProfiler-GEMM

> 一个单文件、零第三方依赖的 GEMM（矩阵乘法）体系结构优化实验：从朴素 `i-j-k`
> 三重循环出发，逐步加入**循环重排**、**Cache Line 对齐**、**向量化 / FMA 融合**、
> **矩阵分块（Cache Blocking）** 与 **AVX2 寄存器分块微内核**，把大矩阵
> （`8192×8192`）下的 GFLOPS 从断崖式下跌的 **6.2** 拉回 **19.7**（约 **3.2×**），
> 并保持单文件、命令行可配置的易用性。

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
- **校验**：分块/微内核结果与 `i-k-j` 结果做**全矩阵相对误差**比对，另与独立点积做抽样比对，保证每一阶段数值正确。

优化策略按五个 Phase 递进，每个 Phase 保持数值结果一致，只改变访存模式与编译方式。

---

## 2. 核心结果

在 Intel Core Ultra 7 255H 上（单线程，GCC 15.2 `-O3 -march=native -ffast-math`）：

| 阶段 | 实现 | 1024×1024 | 4096×4096 | 8192×8192 |
|------|------|----------:|----------:|----------:|
| Phase 1 | `i-j-k` 朴素 | 0.84 GFLOPS | — | — |
| Phase 3 | `i-k-j` + 对齐 + AVX2/FMA | 13.1 GFLOPS | 6.4 GFLOPS | **6.2 GFLOPS** |
| Phase 4 | + 矩阵分块 (block=64) | 20.6 GFLOPS | 14.9 GFLOPS | 11.0 GFLOPS |
| Phase 5 | + AVX2 寄存器分块微内核 | **37.4 GFLOPS** | **26.0 GFLOPS** | **19.7 GFLOPS** |

> 关键观察：未分块的 `i-k-j` 在矩阵从 `1024²` 增大到 `8192²` 时，GFLOPS 从 13.1
> 跌到 6.2（容量/冲突 Miss 与 TLB 压力导致“断崖式下跌”）；加入**矩阵分块**后恢复
> 到 11.0，再叠加**寄存器分块微内核**后达到 19.7 GFLOPS——**已经高于未优化时
> 1024² 的吞吐**，断崖被消除。

---

## 3. 硬件与环境

| 项目 | 配置 |
|------|------|
| CPU | Intel Core Ultra 7 255H（Arrow Lake） |
| 主机 OS | Windows 11 Home（24H2） |
| 编译器 | GCC 15.2.0（MinGW-W64 x86_64-ucrt-posix-seh） |
| 构建系统 | CMake ≥ 3.16 |
| 语言标准 | C++17（`-std=c++17`，禁用 GNU 扩展） |
| 优化标志 | `-O3 -march=native -ffast-math`（MSVC 分支：`/O2 /arch:AVX2 /fp:fast`） |
| 向量化 | AVX2 + FMA（微内核 `MR=8 × NR=4`） |

---

## 4. 环境依赖

- **CMake** ≥ 3.16
- **GCC / Clang**（支持 C++17 与 `-march=native`；MSVC 亦有等价分支）
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

# 自动扫描 block 大小并给出建议 (只对 Phase 5 有效)
./build/gemm 4096 64 --tune
```

### Debug

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/gemm 512
```

### 命令行参数

| 参数 | 含义 | 默认 |
|------|------|------|
| `argv[1]` | 方阵维度 `N` | 1024 |
| `argv[2]` | 分块大小 `block` | 64 |
| `--tune` | 扫描 `{32,48,64,96,128,160,192,256}` 并给出最优 block | 关闭 |

> `-march=native` 会针对**当前 CPU** 生成指令，换机器可能触发 `Illegal instruction`；
> 如需可移植产物，改用 `-march=x86-64-v3`（AVX2）或 `-march=x86-64-v4`（AVX-512）。
> 在无 AVX2/FMA 的机器上，程序会自动退化为 Phase 4 的标量分块实现。

---

## 6. 实现演进

### Phase 1 — Baseline（`i-j-k`）

```cpp
for (i) for (j) for (k)
    C[i*N+j] += A[i*N+k] * B[k*N+j];
```

最内层 `k` 循环访问 `B[k*N+j]`：相邻两次访问间隔 `N × 8B`，彻底丧失空间局部性。
当 `N` 为 2 的幂（如 1024）时，8 KB 的行跨距还会命中 Cache Set 冲突，进一步恶化。

### Phase 2/3 — 循环重排（`i-k-j`）+ 对齐 & SIMD

```cpp
for (i) for (k) {
    const double a = A[i*N+k];
    for (j) C[i*N+j] += a * B[k*N+j];
}
```

- 最内层 `j` 循环访问 `B[k*N+j]`：连续 8 个 `double` 落在同一条 64B Cache Line 上，实现 Stride-1；
- **64B 对齐**：`malloc` 多分配 `ALIGN-1+sizeof(void*)`，原始指针存于对齐地址之前，`free` 释放原始指针；
- **向量化 / FMA**：`-march=native` 激活 AVX2/FMA，`a*b+c` 融合为 `vfmadd231pd`，配合 `-ffast-math` 解锁重排。

### Phase 4 — 矩阵分块（Cache Blocking / Tiling）

```cpp
for (i0) for (j0) for (k0)
    for (i in i0..i0+B) for (k in k0..k0+B)
        for (j in j0..j0+B)
            C[i][j] += A[i][k] * B[k][j];
```

把矩阵切成 `block×block` 的 tile。循环顺序 `i0 → j0 → k0` 保证：对固定 `(i0, j0)`，
`C` 的 `block×block` tile 在整个 `k0` 循环期间常驻 cache，`A`/`B` 每次只搬入一个
`block×block` tile。工作集从 `O(N)` 降为 `O(block²)`，`B` 的重复 DRAM 读取与 TLB
压力都大幅下降，是消除大矩阵 GFLOPS 断崖的核心一步。

### Phase 5 — AVX2 寄存器分块微内核

```cpp
// MR=8 行 × NR=4 列 (一个 YMM = 4 double)
for (k in k0..k_end) {
    __m256d bv = load(B[k*N+j .. j+4]);
    for (r in 0..MR) {
        __m256d av = broadcast(A[(i+r)*N+k]);
        acc[r] = _mm256_fmadd_pd(av, bv, acc[r]);
    }
}
```

在分块内部再取 `MR×NR` 小块，`MR×NR` 个累加器常驻 YMM 寄存器，`C` 只在进出微内核时
load/store 一次；每个 FMA 真正贡献计算，消除了标量分块里 `C` 反复 load/store 的冗余。
`MR=8` 时一次 `B` 向量 load 被 8 个 FMA 共享。

---

## 7. 性能对比

| 阶段 | 循环顺序 | 优化手段 | 1024² GFLOPS | 8192² GFLOPS |
|------|---------|---------|-------------:|-------------:|
| Phase 1 | `i-j-k` | 无 | 0.84 | —（太慢，跳过） |
| Phase 3 | `i-k-j` | + 64B 对齐 + AVX2 + FMA | 13.1 | 6.2 |
| Phase 4 | `i-k-j` 分块 | + Cache Blocking(64) | 20.6 | 11.0 |
| Phase 5 | 微内核 | + AVX2 寄存器分块 (8×4) | 37.4 | 19.7 |

> 数据为同一机器上单次运行的典型值，受睿频 / 温度影响会有 ±5% 左右波动。
> 8192² 时 `Phase5 / Phase3 ≈ 3.2×`；更重要的是绝对吞吐 19.7 GFLOPS 已超过
> 未优化时 1024² 的 13.1 GFLOPS，说明断崖被有效消除。

**加速来源分解**：

- 循环重排：把 `B` 的跨行随机访存变为 Stride-1 顺序访存（约 8~15×）；
- Cache Blocking：把 `B` 的重复 DRAM 读取从 `N³` 降到 `N³/block`，消除容量 Miss 与 TLB 压力；
- 寄存器分块：把 `C` 的重复 load/store 从 `N³` 降到 `N²`，让 FMA 占比最大化；
- AVX2+FMA：4 个 double 打包 + 乘加融合，单条指令完成 8 FLOP。

---

## 8. 体系结构优化原理

### 8.1 空间局部性与 Cache Line

`B[k*N+j]` 连续访问时，8 个 `double`（64B）共享一条 Cache Line，空间局部性 100% 被利用；
反之 `i-j-k` 的内层跨行访问一次只用一个 Cache Line，浪费带宽。

### 8.2 64 字节对齐

消除跨 Cache Line 的首行访问分裂，并允许编译器发射对齐向量加载 `vmovapd`（而非 `vmovupd`）。

### 8.3 AVX2 向量化与 FMA

- **AVX2**：256-bit 寄存器一次处理 4 个 `double`；
- **FMA**：`乘 + 加` 融合为一条指令，指令数减半且只做一次舍入；
- 两者叠加后，Phase 3 相对纯重排再提速，Phase 5 用显式 intrinsic 把 FMA 占比推到最大。

### 8.4 Cache Blocking 如何消除 8192² 断崖

`8192×8192` 的 `B` 约 512 MB，远超 L2/L3。未分块的 `i-k-j` 对每个 `i` 都要把整个 `B`
重新读一遍（总 `B` 流量约 `N³×8B ≈ 4 TB`），并且每行 64 KB 跨越 16 个 4 KB 页，
导致大量 TLB Miss——这就是 GFLOPS 断崖的来源。分块后 `B` 以 `block×block` tile
（32 KB）为单位在 `i` 块间复用，`B` 流量降到 `N³/block×8B ≈ 69 GB`，TLB 压力同步下降，
吞吐从 6.2 恢复到 11.0，再经寄存器分块达到 19.7。

### 8.5 `-ffast-math` 的作用

严格 IEEE-754 语义禁止重排/融合浮点运算，会阻断自动向量化与 FMA。`-ffast-math`
放宽约束，解锁向量化与乘加融合，代价是结果与严格顺序相差数个 ULP——这也是校验改用
**相对误差**而非精确 `==` 的原因。

---

## 9. 正确性校验

程序每次运行都做两层校验：

1. **全矩阵比对**：`C_tiled`、`C_avx2` 分别与 `C_ikj` 逐元素比较最大相对误差；
2. **独立抽样**：随机取 16 个元素，与独立的点积 `Σ A[i][k]·B[k][j]` 比对。

示例输出（`N=8192`）：

```
全矩阵校验 C_tiled vs C_ikj : max 相对误差 = 0.000e+00  OK
全矩阵校验 C_avx2  vs C_ikj : max 相对误差 = 0.000e+00  OK
抽样校验 (16 个随机元素 vs 独立点积): max 相对误差 = 1.624e-14  OK
```

分块/微内核与 `i-k-j` 对每个 `C[i][j]` 都按 `k` 升序累加、且同样使用 FMA，故结果**逐位一致**；
与独立点积的差异在 `1e-14` 量级，属正常浮点噪声。断言阈值 `1e-9` 远大于噪声、远小于可观测误差。

---

## 10. WSL2 的 PMU 限制

在 WSL2 中尝试抓取硬件计数器：

```bash
perf stat -e L1-dcache-load-misses,instructions,cycles ./build/gemm
```

结果报错 `Error: No supported events found.`。机制分析：

1. WSL2 运行在 Hyper-V 之上，宿主 Hypervisor 拥有 PMU 完全控制权，默认不将硬件计数器虚拟化透传给客户机；
2. `perf list` 仅列出 software/tracepoint/kprobe/msr 等软件事件；`/sys/bus/event_source/devices/` 下缺少 `cpu` PMU 设备；
3. 这不是 `perf_event_paranoid` 权限问题，而是虚拟化层的架构限制。

**替代路径**：Windows 原生 + Intel VTune（`INST_RETIRED.ANY` / `CPU_CLK_UNHALTED.THREAD` / `L1D.REPLACEMENT`）、原生 Linux、或 `-cpu host` 直通 PMU 的 KVM 虚拟机。

---

## 11. 后续优化方向

当前 19.7 GFLOPS（8192²）距该 CPU 理论单核峰值仍有差距，可继续展开：

1. **Packing（数据重排）**：把 `A`/`B` 的 tile 拷贝到连续缓冲，微内核里 `A` 也改为向量 load、`B` 顺序复用，进一步压缩访存指令占比；
2. **多线程（OpenMP）**：按 `i0` 行块并行，利用多核带宽（16 逻辑核）；
3. **更大微内核 / AVX-512**：`MR=16, NR=8` 或 512-bit 向量，摊薄循环开销；
4. **Roofline 定位**：实测内存带宽与峰值算力，判断当前瓶颈在访存还是计算。

---

## 12. 文件结构

```
TinyProfiler-GEMM/
├── CMakeLists.txt   # 构建脚本 (-O3 -march=native -ffast-math)
├── main.cpp         # GEMM 单文件实现 (5 个 Phase + 计时 + 双层校验)
├── README.md        # 本实验报告
└── .gitignore       # 忽略构建产物
```

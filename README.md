# TinyProfiler-GEMM

> 1024×1024 矩阵乘法（GEMM）的体系结构优化实验：从朴素 `i-j-k` 三重循环出发，
> 通过**循环重排**、**Cache Line 对齐**与**向量化 / FMA 融合**，将吞吐从
> **1.477 GFLOPS 提升到 12.144 GFLOPS**（约 **8.2×**），并记录 WSL2 虚拟化下
> `perf` 硬件计数器受限的机制。

---

## 目录

1. [项目简介](#1-项目简介)
2. [硬件与环境](#2-硬件与环境)
3. [环境依赖](#3-环境依赖)
4. [构建与运行](#4-构建与运行)
5. [实现演进](#5-实现演进)
6. [性能对比](#6-性能对比)
7. [体系结构优化原理分析](#7-体系结构优化原理分析)
8. [正确性校验](#8-正确性校验)
9. [系统分析：WSL2 的 PMU 限制](#9-系统分析wsl2-的-pmu-限制)
10. [后续优化方向](#10-后续优化方向)
11. [文件结构](#11-文件结构)

---

## 1. 项目简介

本项目实现了一个单文件、零第三方依赖的 GEMM（General Matrix Multiply）baseline：

- **计算**：`C = A × B`，`A/B/C` 均为 `1024×1024` 的 `double` 方阵（`M = K = N = 1024`）；
- **存储**：行主序（row-major），`A[i][k] = A[i * N + k]`；
- **计时**：`std::chrono::steady_clock` + `duration<double, milli>`，毫秒级（含小数）；
- **校验**：重算 `C[0][0]` 并做相对误差断言，保证每一阶段优化后数值正确；
- **指标**：`GFLOPS = 2·N³ / (t_ms·1e-3) / 1e9`，其中 `2·N³` 为乘加各算一次的浮点运算总数。

优化策略按三个 Phase 递进，每个 Phase 保持数值结果一致，只改变访存模式与编译方式。

---

## 2. 硬件与环境

| 项目 | 配置 |
|------|------|
| CPU | Intel Core Ultra 7 255H（Arrow Lake） |
| 主机 OS | Windows 11 Home（24H2） |
| 虚拟化 | WSL2（Ubuntu 26.04，内核 `6.6.114.1-microsoft-standard-WSL2`） |
| 编译器 | GCC 15.2.0（MinGW-W64 x86_64-ucrt-posix-seh / g++ 15.2.0） |
| 构建系统 | CMake ≥ 3.16 |
| 语言标准 | C++17（`-std=c++17`，禁用 GNU 扩展） |
| 优化标志 | 默认 `-O3`；Phase 3 追加 `-march=native -ffast-math` |

---

## 3. 环境依赖

- **CMake** ≥ 3.16
- **GCC / Clang**（支持 C++17 与 `-march=native`；MSVC 亦有等价分支 `/O2 /arch:AVX2 /fp:fast`）
- （可选）**Linux `perf`** —— 硬件计数器分析，见 [第 9 节](#9-系统分析wsl2-的-pmu-限制)
- （可选）**Intel VTune Profiler** —— Windows 下的硬件计数器替代方案

---

## 4. 构建与运行

### Release（推荐，`-O3` + `-march=native -ffast-math`）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Linux / WSL2
./build/gemm

# Windows (MinGW 产物带 .exe 后缀)
./build/gemm.exe
```

### Debug（`-O0`，保留调试信息）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/gemm
```

> 说明：`-march=native` 会针对**当前 CPU** 生成指令（如 AVX2 / FMA），
> 换机器运行可能触发 `Illegal instruction`。若要发布到其他机器，请改用固定目标
> `-march=x86-64-v3`（AVX2）或 `-march=x86-64-v4`（AVX-512）。

---

## 5. 实现演进

### Phase 1 — Baseline（`i-j-k`）

```cpp
for (i) for (j) for (k)
    C[i*N+j] += A[i*N+k] * B[k*N+j];
```

最内层 `k` 循环访问 `B[k*N+j]`：相邻两次访问间隔 `N × 8B = 8 KB`，彻底丧失空间局部性。

### Phase 2 — Cache 重构（`i-k-j`）

```cpp
for (i) for (k) {
    const double a = A[i*N+k];
    for (j)
        C[i*N+j] += a * B[k*N+j];
}
```

最内层 `j` 循环访问 `B[k*N+j]`：连续 8 个 `double` 落在同一条 64B Cache Line 上，
实现 Stride-1 顺序访存，`C` 的第 `i` 行全程常驻 L1。

### Phase 3 — 对齐 & SIMD（`i-k-j` + 64B 对齐 + AVX2 + `-ffast-math`）

- **64B 对齐**：放弃 `std::vector` 默认分配，改为 Heap 上 64 字节对齐的连续内存
  （Windows/MinGW 无 `std::aligned_alloc`/`posix_memalign`，采用经典手动对齐：
  `malloc` 多分配 `ALIGN-1+sizeof(void*)`，原始指针存于对齐地址之前，`free` 释放原始指针）；
- **向量化**：`-march=native` 显式激活 AVX2/FMA；
- **FMA 融合**：`a*b+c` 融合为单条 `vfmadd231pd` 乘加指令，并启用 `-ffast-math`。

---

## 6. 性能对比

| 阶段 | 循环顺序 | 优化手段 | 耗时 (ms) | GFLOPS | 相对加速 |
|------|---------|---------|----------:|-------:|--------:|
| Phase 1 | `i-j-k` | 无（Baseline） | 1454.015 | 1.477 | 1.00× |
| Phase 2 | `i-k-j` | 循环重排（Stride-1 顺序访存） | 210.537 | 10.200 | 6.91× |
| Phase 3 | `i-k-j` | + 64B 对齐 + AVX2 + FMA（`-ffast-math`） | 176.838 | 12.144 | **8.22×** |

> 注：`2·N³ = 2·1024³ ≈ 2.147 GFLOP`，各阶段 `耗时 × GFLOPS ≈ 2.147`，数据自洽。

**加速分解**：循环重排贡献 ~6.9×（访存模式质变），对齐 + 向量化 + FMA 再贡献 ~1.19×。

---

## 7. 体系结构优化原理分析

### 7.1 循环重排 `i-j-k` → `i-k-j`（核心加速，~6.9×）

`i-j-k` 的致命缺陷在于对 `B` 的**按列跨步访问**：

- 最内层 `k` 循环中，`B[k*N+j]` 的地址步长为 `N × sizeof(double) = 8 KB`；
- 一条 64B Cache Line 只容纳 8 个 `double`，而 8 KB 的步长意味着**每取一个元素都要换一条新 Cache Line**，
  单次访存只利用了所取 64B 中的 8B（利用率仅 12.5%，其余 7/8 的带宽被浪费）；
- 更糟的是 `k` 循环横扫整个 8 MB 的 `B` 矩阵，远超 L1d（~48 KB）/L2 容量，
  大量行在回访之前已被逐出——**空间局部性与时间局部性同时被击穿**，程序沦为内存带宽瓶颈。

`i-k-j` 将 `j` 提为最内层循环：

- `B[k*N+j]` 随 `j` 递增按 Stride-1 连续访问，**连续 8 个 `double` 共享同一 Cache Line**，
  空间局部性被 100% 利用；
- `C[i*N+j]` 的第 `i` 行（8 KB）在整个 `k` 循环期间常驻 L1，`A[i*N+k]` 则被提到中间层循环外，
  作为标量复用，`k` 循环的写回命中率极高。

这是 BLAS 级别优化的第一个、也是最廉价有效的一步：**仅改变循环嵌套顺序，即可获得近 7× 提升**。

### 7.2 64 字节内存对齐

- **消除跨 Cache Line 访问**：若矩阵行首地址未对齐到 64B，一行数据可能横跨两条 Cache Line，
  导致本应 1 次命中的行首访存分裂为 2 次；
- **启用对齐向量加载**：对齐后的指针允许编译器发射 `vmovapd`（对齐加载）而非更慢的 `vmovupd`（非对齐加载）；
- 本实验在三块矩阵上验证首地址 `% 64 == 0`（末字节为 `0x80`），全部 `OK`。

### 7.3 AVX2 向量化与 FMA 乘加融合

- **AVX2**：256-bit 向量寄存器一次处理 4 个 `double`，理论指令数降为标量的 1/4；
- **FMA（`vfmadd231pd`）**：把 `乘 + 加` 两条指令融合为一条，指令数减半的同时**只做一次舍入**，
  精度优于先乘后加，带宽效率更高；
- 两者叠加后，Phase 3 相比 Phase 2 再提速约 19%。

### 7.4 `-ffast-math` 的作用

严格 IEEE-754 语义禁止编译器重排/融合浮点运算，会**阻断自动向量化与 FMA 生成**。
`-ffast-math` 放宽了这些约束（允许结合律重排、假设无 NaN/Inf、忽略符号零），
从而解锁向量化与乘加融合。代价是结果可能与严格顺序相差数个 ULP——这也是第 8 节改用
**相对误差断言**而非精确 `==` 的原因。

---

## 8. 正确性校验

每个 Phase 都重算 `C[0][0] = Σ A[0][k]·B[k][0]` 作为参考值，并与矩阵乘法结果比较：

```
校验 C[0][0]: 计算值 = 328.782400, 参考值 = 328.782400, 相对误差 = 3.458e-16, OK
```

- **相对误差 `3.458e-16`**，与 `double` 的机器精度 `ε ≈ 2.22e-16` 同量级，属 FMA 融合的正常舍入差异；
- 断言阈值 `rel_err < 1e-9`，远大于浮点噪声、远小于可观测误差，稳健且不失真。

---

## 9. 系统分析：WSL2 的 PMU 限制

在 WSL2 中尝试抓取硬件计数器：

```bash
perf stat -e L1-dcache-load-misses,instructions,cycles ./build/gemm
```

结果报错：

```
Error: No supported events found.
The L1-dcache-load-misses:u event is not supported.
```

**机制分析**：

1. **Hyper-V 独占硬件 PMU**：WSL2 运行在 Hyper-V 虚拟化之上，宿主 Hypervisor 拥有对
   Performance Monitoring Unit（PMU）的完全控制权，**默认不将硬件性能计数器虚拟化透传给客户机**；
2. **证据链**：
   - `perf list` 仅列出 `software` / `tracepoint` / `kprobe` / `msr` 等**软件事件**，无任何硬件事件；
   - `/sys/bus/event_source/devices/` 下**缺少 `cpu` PMU 设备**（只有 `breakpoint/kprobe/msr/software/tracepoint/uprobe`）；
   - 连最基础的 `instructions`、`cycles` 也返回 `not supported`；
3. **结论**：这不是 `perf_event_paranoid` 权限问题，也不是包/配置能修复的——是虚拟化层的架构限制。
   硬件事件 `L1-dcache-load-misses / instructions / cycles` 在 WSL2 中**不可获取**。

**可用的替代路径**：

| 方案 | 能否拿到硬件计数器 | 说明 |
|------|:---:|------|
| Windows 原生 + Intel VTune | ✅ | `INST_RETIRED.ANY` / `CPU_CLK_UNHALTED.THREAD` / `L1D.REPLACEMENT` 一一对应 |
| 原生 Linux（双系统/独立机器） | ✅ | `perf stat` 完全可用 |
| KVM 虚拟机（`-cpu host` 直通 PMU） | ✅ | 需非 Hyper-V 的虚拟化方案 |

> 软件事件仍可采集（`task-clock` / `context-switches` / `page-faults` 等），
> 用于观察调度与缺页行为，但无法反映缓存与指令级微架构行为。

---

## 10. 后续优化方向

当前 12.144 GFLOPS 距离该 CPU 的理论峰值仍有数量级差距，后续 Phase 可依次展开：

1. **Cache Blocking / Tiling**：将矩阵切分为 `64×64×64` 的 tile，使 `A`/`B`/`C` 的
   工作集同时驻留 L1/L2，进一步降低容量型 Cache Miss；
2. **显式 SIMD Intrinsics**：手写 `_mm256_fmadd_pd` / AVX-512 内联汇编，突破编译器自动向量化的局限；
3. **多线程（OpenMP）**：按 `i` 行或 tile 并行，利用多核带宽；
4. **Register Blocking / Packing**：重排矩阵存储布局，消除 TLB 压力并最大化寄存器复用；
5. **Roofline 模型定位**：测量实际内存带宽与峰值算力，判断当前瓶颈在访存还是计算。

---

## 11. 文件结构

```
TinyProfiler-GEMM/
├── CMakeLists.txt   # 构建脚本（含 -march=native -ffast-math）
├── main.cpp         # GEMM 单文件实现（i-k-j + 64B 对齐 + chrono 计时 + 相对误差校验）
├── README.md        # 本实验报告
└── .gitignore       # 忽略构建产物
```

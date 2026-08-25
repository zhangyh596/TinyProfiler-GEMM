// GEMM Phase 4/5: Cache Blocking (矩阵分块) + AVX2 寄存器分块微内核
// 在 Phase 3 (i-k-j + 64B 对齐 + AVX2/FMA + -ffast-math) 基础上:
//   Phase 4  将 N×N 矩阵按 block×block 分块, 使 A/B/C 的 tile 常驻 cache,
//            大幅减少大矩阵 (如 8192×8192) 下 B 的重复 DRAM 读取与 TLB 压力。
//   Phase 5  在分块内部再用 MR×NR 寄存器分块 (AVX2 _mm256_fmadd_pd),
//            把 C 的 MR×NR 小块常驻 YMM 寄存器, 消除 C 的重复 load/store,
//            让每个 FMA 都真正贡献到计算吞吐。
//
// 用法:
//   gemm [N] [block] [--tune]
//     N      方阵维度, 默认 1024
//     block  分块大小, 默认 64
//     --tune 自动扫描一组 block 大小并给出建议
//
// 构建:
//   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
//   ./build/gemm 8192 64
//
// 注: Windows/MinGW 无 std::aligned_alloc / posix_memalign,
//     故采用经典手动对齐: malloc 多分配, 原始指针存于对齐地址之前, free 释放原始指针。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define GEMM_HAS_AVX2_FMA 1
#endif

namespace {

constexpr std::size_t ALIGN = 64;  // 对齐到 64 字节 (CPU Cache Line)

// 分配 n 个 double, 返回 64 字节对齐指针 (失败返回 nullptr)
double* alloc_aligned64(std::size_t n) {
    const std::size_t bytes = n * sizeof(double);
    void* raw = std::malloc(bytes + ALIGN - 1 + sizeof(void*));
    if (!raw) return nullptr;
    const std::uintptr_t aligned =
        (reinterpret_cast<std::uintptr_t>(raw) + sizeof(void*) + ALIGN - 1) &
        ~static_cast<std::uintptr_t>(ALIGN - 1);
    reinterpret_cast<void**>(aligned)[-1] = raw;
    return reinterpret_cast<double*>(aligned);
}

void free_aligned64(double* p) {
    if (p) std::free(reinterpret_cast<void**>(p)[-1]);
}

bool is_aligned64(const double* p) {
    return reinterpret_cast<std::uintptr_t>(p) % ALIGN == 0;
}

void init_matrices(std::size_t n, double* A, double* B) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            A[i * n + k] = static_cast<double>((i * n + k) % 100) * 0.01;
            B[i * n + k] = static_cast<double>((k * n + i) % 100) * 0.01;
        }
    }
}

// Phase 1: 朴素 i-j-k (baseline / 参考实现)
void gemm_naive(std::size_t n, const double* A, const double* B, double* C) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Phase 2/3: i-k-j (B 按行连续访问, C 第 i 行常驻 cache)
void gemm_ikj(std::size_t n, const double* __restrict A,
              const double* __restrict B, double* __restrict C) {
    for (std::size_t i = 0; i < n; ++i) {
        double* Ci = C + i * n;
        for (std::size_t k = 0; k < n; ++k) {
            const double a = A[i * n + k];
            const double* Bk = B + k * n;
            for (std::size_t j = 0; j < n; ++j) {
                Ci[j] += a * Bk[j];
            }
        }
    }
}

// Phase 4: Cache Blocking (矩阵分块), 块内仍为 i-k-j 标量形式
// 循环顺序 i0 -> j0 -> k0: 对固定 (i0, j0), C 的 block×block tile 在整个
// k0 循环期间常驻 L1/L2; 每次只搬入 B 与 A 的 block×block tile。
void gemm_tiled(std::size_t n, const double* __restrict A,
                const double* __restrict B, double* __restrict C,
                std::size_t block) {
    if (block == 0 || block > n) block = n;
    for (std::size_t i0 = 0; i0 < n; i0 += block) {
        const std::size_t i_end = std::min(i0 + block, n);
        for (std::size_t j0 = 0; j0 < n; j0 += block) {
            const std::size_t j_end = std::min(j0 + block, n);
            const std::size_t jlen = j_end - j0;
            for (std::size_t k0 = 0; k0 < n; k0 += block) {
                const std::size_t k_end = std::min(k0 + block, n);
                for (std::size_t i = i0; i < i_end; ++i) {
                    double* Ci = C + i * n + j0;
                    const double* Ai = A + i * n;
                    for (std::size_t k = k0; k < k_end; ++k) {
                        const double a = Ai[k];
                        const double* Bk = B + k * n + j0;
                        for (std::size_t j = 0; j < jlen; ++j) {
                            Ci[j] += a * Bk[j];
                        }
                    }
                }
            }
        }
    }
}

// Phase 5: Cache Blocking + AVX2 寄存器分块微内核
// MR×NR 个累加器常驻 YMM 寄存器, 内层沿 k 做 FMA, C 只在进出微内核时
// load/store 一次, 极大降低访存指令占比。
#if GEMM_HAS_AVX2_FMA
constexpr std::size_t MR = 8;  // 微内核一次处理的行数
constexpr std::size_t NR = 4;  // 微内核一次处理的列数 (一个 256-bit 向量 = 4 double)

void gemm_tiled_avx2(std::size_t n, const double* __restrict A,
                     const double* __restrict B, double* __restrict C,
                     std::size_t block) {
    if (block == 0 || block > n) block = n;
    for (std::size_t i0 = 0; i0 < n; i0 += block) {
        const std::size_t i_end = std::min(i0 + block, n);
        for (std::size_t j0 = 0; j0 < n; j0 += block) {
            const std::size_t j_end = std::min(j0 + block, n);
            for (std::size_t k0 = 0; k0 < n; k0 += block) {
                const std::size_t k_end = std::min(k0 + block, n);
                const std::size_t ibulk = i_end - ((i_end - i0) % MR);
                const std::size_t jbulk = j_end - ((j_end - j0) % NR);

                // 主体: MR×NR 微内核, 覆盖 [i0, ibulk) × [j0, jbulk)
                for (std::size_t i = i0; i < ibulk; i += MR) {
                    for (std::size_t j = j0; j < jbulk; j += NR) {
                        __m256d acc[MR];
                        for (std::size_t r = 0; r < MR; ++r) {
                            acc[r] = _mm256_loadu_pd(&C[(i + r) * n + j]);
                        }
                        for (std::size_t k = k0; k < k_end; ++k) {
                            const __m256d bv =
                                _mm256_loadu_pd(&B[k * n + j]);
                            for (std::size_t r = 0; r < MR; ++r) {
                                const __m256d av =
                                    _mm256_broadcast_sd(&A[(i + r) * n + k]);
                                acc[r] = _mm256_fmadd_pd(av, bv, acc[r]);
                            }
                        }
                        for (std::size_t r = 0; r < MR; ++r) {
                            _mm256_storeu_pd(&C[(i + r) * n + j], acc[r]);
                        }
                    }
                }

                // 余数: 主体行 [i0, ibulk) 中剩下的列 [jbulk, j_end)
                for (std::size_t i = i0; i < ibulk; ++i) {
                    double* Ci = C + i * n;
                    const double* Ai = A + i * n;
                    for (std::size_t k = k0; k < k_end; ++k) {
                        const double a = Ai[k];
                        const double* Bk = B + k * n;
                        for (std::size_t j = jbulk; j < j_end; ++j) {
                            Ci[j] += a * Bk[j];
                        }
                    }
                }

                // 余数: 剩下的行 [ibulk, i_end) 整行
                for (std::size_t i = ibulk; i < i_end; ++i) {
                    double* Ci = C + i * n;
                    const double* Ai = A + i * n;
                    for (std::size_t k = k0; k < k_end; ++k) {
                        const double a = Ai[k];
                        const double* Bk = B + k * n;
                        for (std::size_t j = j0; j < j_end; ++j) {
                            Ci[j] += a * Bk[j];
                        }
                    }
                }
            }
        }
    }
}
#endif  // GEMM_HAS_AVX2_FMA

double gflops_of(std::size_t n, double ms) {
    return 2.0 * static_cast<double>(n) * n * n / (ms * 1e-3) / 1e9;
}

double max_rel_err(std::size_t n, const double* a, const double* b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < n * n; ++i) {
        const double ref = b[i];
        const double denom = std::fabs(ref) > 1e-30 ? std::fabs(ref) : 1e-30;
        const double e = std::fabs(a[i] - ref) / denom;
        if (e > worst) worst = e;
    }
    return worst;
}

double dot_ref(std::size_t n, const double* A, const double* B,
               std::size_t row, std::size_t col) {
    double s = 0.0;
    for (std::size_t k = 0; k < n; ++k) s += A[row * n + k] * B[k * n + col];
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t N = 1024;
    std::size_t BLOCK = 64;
    bool tune = false;

    if (argc > 1) {
        const long v = std::strtol(argv[1], nullptr, 10);
        if (v > 0) N = static_cast<std::size_t>(v);
    }
    if (argc > 2) {
        const long v = std::strtol(argv[2], nullptr, 10);
        if (v > 0) BLOCK = static_cast<std::size_t>(v);
    }
    for (int a = 1; a < argc; ++a) {
        if (std::strcmp(argv[a], "--tune") == 0) tune = true;
    }

    const double bytes_per_mat = static_cast<double>(N) * N * sizeof(double);
    std::printf("============================================================\n");
    std::printf(" GEMM 性能测试  N = %zu  (单矩阵 %.1f MB)\n", N,
                bytes_per_mat / 1e6);
#if GEMM_HAS_AVX2_FMA
    std::printf(" 向量化: AVX2 + FMA (微内核 MR=%zu x NR=%zu)\n", MR, NR);
#endif
    std::printf("============================================================\n");

    double* A = alloc_aligned64(N * N);
    double* B = alloc_aligned64(N * N);
    double* C_ikj = alloc_aligned64(N * N);
    double* C_tiled = alloc_aligned64(N * N);
    double* C_avx2 = alloc_aligned64(N * N);
    if (!A || !B || !C_ikj || !C_tiled || !C_avx2) {
        std::printf("内存分配失败\n");
        free_aligned64(A);
        free_aligned64(B);
        free_aligned64(C_ikj);
        free_aligned64(C_tiled);
        free_aligned64(C_avx2);
        return 1;
    }

    init_matrices(N, A, B);

    std::printf("64 字节对齐校验: A=%s B=%s C_ikj=%s C_tiled=%s C_avx2=%s\n",
                is_aligned64(A) ? "OK" : "FAIL",
                is_aligned64(B) ? "OK" : "FAIL",
                is_aligned64(C_ikj) ? "OK" : "FAIL",
                is_aligned64(C_tiled) ? "OK" : "FAIL",
                is_aligned64(C_avx2) ? "OK" : "FAIL");

    // ---------- Phase 3: i-k-j (未分块) ----------
    std::fill(C_ikj, C_ikj + N * N, 0.0);
    double ms_ikj = 0.0;
    {
        const auto t0 = std::chrono::steady_clock::now();
        gemm_ikj(N, A, B, C_ikj);
        const auto t1 = std::chrono::steady_clock::now();
        ms_ikj = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("[Phase 3] i-k-j (未分块)             : %9.1f ms  (%.3f GFLOPS)\n",
                    ms_ikj, gflops_of(N, ms_ikj));
    }

    // ---------- Phase 4: Cache Blocking ----------
    std::fill(C_tiled, C_tiled + N * N, 0.0);
    double ms_tiled = 0.0;
    {
        const auto t0 = std::chrono::steady_clock::now();
        gemm_tiled(N, A, B, C_tiled, BLOCK);
        const auto t1 = std::chrono::steady_clock::now();
        ms_tiled = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("[Phase 4] i-k-j + Blocking(%4zu)     : %9.1f ms  (%.3f GFLOPS)\n",
                    BLOCK, ms_tiled, gflops_of(N, ms_tiled));
    }

    // ---------- Phase 5: Cache Blocking + AVX2 寄存器分块 ----------
    double ms_avx2 = 0.0;
    std::fill(C_avx2, C_avx2 + N * N, 0.0);
#if GEMM_HAS_AVX2_FMA
    {
        const auto t0 = std::chrono::steady_clock::now();
        gemm_tiled_avx2(N, A, B, C_avx2, BLOCK);
        const auto t1 = std::chrono::steady_clock::now();
        ms_avx2 = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("[Phase 5] + AVX2 寄存器分块微内核    : %9.1f ms  (%.3f GFLOPS)\n",
                    ms_avx2, gflops_of(N, ms_avx2));
    }
#else
    std::printf("[Phase 5] + AVX2 寄存器分块微内核    : 不可用 (需 -march=native/AVX2)\n");
#endif

    // ---------- 正确性校验 ----------
    const double err4 = max_rel_err(N, C_tiled, C_ikj);
    std::printf("------------------------------------------------------------\n");
    std::printf("全矩阵校验 C_tiled vs C_ikj : max 相对误差 = %.3e  %s\n", err4,
                (err4 < 1e-9) ? "OK" : "FAIL");
#if GEMM_HAS_AVX2_FMA
    const double err5 = max_rel_err(N, C_avx2, C_ikj);
    std::printf("全矩阵校验 C_avx2  vs C_ikj : max 相对误差 = %.3e  %s\n", err5,
                (err5 < 1e-9) ? "OK" : "FAIL");
#endif

    // 抽样校验: 与独立点积参考值对比 (覆盖任意 N, 复杂度 O(N)/元素)
    {
        std::mt19937_64 rng(20260825ULL);
        std::uniform_int_distribution<std::size_t> dist(0, N - 1);
        double werr = 0.0;
#if GEMM_HAS_AVX2_FMA
        const double* Cchk = C_avx2;
#else
        const double* Cchk = C_tiled;
#endif
        for (int s = 0; s < 16; ++s) {
            const std::size_t r = dist(rng), c = dist(rng);
            const double ref = dot_ref(N, A, B, r, c);
            const double got = Cchk[r * N + c];
            const double denom = std::fabs(ref) > 1e-30 ? std::fabs(ref) : 1e-30;
            werr = std::max(werr, std::fabs(got - ref) / denom);
        }
        std::printf("抽样校验 (16 个随机元素 vs 独立点积): max 相对误差 = %.3e  %s\n",
                    werr, (werr < 1e-9) ? "OK" : "FAIL");
    }

    // ---------- 小矩阵补测 Phase 1 朴素版作为 baseline ----------
    if (N <= 1024) {
        double* C_naive = alloc_aligned64(N * N);
        if (C_naive) {
            std::fill(C_naive, C_naive + N * N, 0.0);
            const auto t0 = std::chrono::steady_clock::now();
            gemm_naive(N, A, B, C_naive);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::printf("[Phase 1] i-j-k (朴素 baseline)      : %9.1f ms  (%.3f GFLOPS)\n",
                        ms, gflops_of(N, ms));
            const double e = max_rel_err(N, C_ikj, C_naive);
            std::printf("全矩阵校验 C_ikj vs C_naive : max 相对误差 = %.3e  %s\n", e,
                        (e < 1e-9) ? "OK" : "FAIL");
            free_aligned64(C_naive);
        }
    } else {
        std::printf("[Phase 1] i-j-k (朴素 baseline)      : 跳过 (N > 1024, 过于缓慢)\n");
    }

    // ---------- 自动调优 block 大小 ----------
#if GEMM_HAS_AVX2_FMA
    if (tune) {
        const std::size_t candidates[] = {32, 48, 64, 96, 128, 160, 192, 256};
        std::size_t best = BLOCK;
        double best_ms = ms_avx2;
        std::printf("------------------------------------------------------------\n");
        std::printf("Block 调优 (Phase 5):\n");
        for (std::size_t b : candidates) {
            if (b > N) continue;
            std::fill(C_avx2, C_avx2 + N * N, 0.0);
            const auto t0 = std::chrono::steady_clock::now();
            gemm_tiled_avx2(N, A, B, C_avx2, b);
            const auto t1 = std::chrono::steady_clock::now();
            const double m =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::printf("  block=%4zu: %9.1f ms  (%.3f GFLOPS)%s\n", b, m,
                        gflops_of(N, m), (m < best_ms) ? "  <-- best" : "");
            if (m < best_ms) {
                best_ms = m;
                best = b;
            }
        }
        std::printf("建议 block = %zu\n", best);
    }
#endif

    std::printf("------------------------------------------------------------\n");
    if (ms_ikj > 0.0 && ms_tiled > 0.0) {
        std::printf("Phase3 -> Phase4 加速比: %.2fx\n", ms_ikj / ms_tiled);
#if GEMM_HAS_AVX2_FMA
        if (ms_avx2 > 0.0) {
            std::printf("Phase3 -> Phase5 加速比: %.2fx\n", ms_ikj / ms_avx2);
        }
#endif
    }

    free_aligned64(A);
    free_aligned64(B);
    free_aligned64(C_ikj);
    free_aligned64(C_tiled);
    free_aligned64(C_avx2);
    return 0;
}


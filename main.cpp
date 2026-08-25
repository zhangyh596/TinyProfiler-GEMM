// GEMM 分层优化实验: i-j-k -> i-k-j -> 对齐/SIMD -> 矩阵分块 -> AVX2 微内核
//                    -> Packing 数据重排 -> OpenMP 多线程。
//
//   Phase 1  i-j-k 朴素 baseline
//   Phase 3  i-k-j + 64B 对齐 + AVX2/FMA + -ffast-math
//   Phase 4  矩阵分块 (Cache Blocking, 块内 i-k-j 标量)
//   Phase 5  分块 + AVX2 寄存器分块微内核 (MR=8 x NR=4)
//   Phase 6  分块 + Packing 数据重排 + 微内核 (MR=4 x NR=8)
//   Phase 7  Phase 6 + OpenMP 多线程 (并行 i0 行块)
//
// 用法:
//   gemm [N] [block] [--tune]
//     N      方阵维度, 默认 1024
//     block  分块大小, 默认 64
//     --tune 自动扫描一组 block 大小 (Phase 6)
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
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define GEMM_HAS_AVX2_FMA 1
#endif

#ifdef _OPENMP
#include <omp.h>
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

// ---------- Phase 1: 朴素 i-j-k ----------
void gemm_naive(std::size_t n, const double* A, const double* B, double* C) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// ---------- Phase 2/3: i-k-j ----------
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

// ---------- Phase 4: Cache Blocking (块内 i-k-j 标量) ----------
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

// ---------- Phase 5: Cache Blocking + AVX2 寄存器分块微内核 ----------
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

                for (std::size_t i = i0; i < ibulk; i += MR) {
                    for (std::size_t j = j0; j < jbulk; j += NR) {
                        __m256d acc[MR];
                        for (std::size_t r = 0; r < MR; ++r) {
                            acc[r] = _mm256_loadu_pd(&C[(i + r) * n + j]);
                        }
                        for (std::size_t k = k0; k < k_end; ++k) {
                            const __m256d bv = _mm256_loadu_pd(&B[k * n + j]);
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

// ---------- Phase 6: Packing 数据重排 + 微内核 (MR=4 x NR=8) ----------
// A 面板按列优先打包 (A_pack[k*MC + r] = A[i0+r][k0+k]), 使固定 k 的 MR 个
// A 元素连续存放 (一个 Cache Line), 微内核里广播 A 只碰 1 条 Cache Line;
// B 面板按行打包, 每行用 memcpy 一次搬入, 微内核里 B 顺序流式复用。
constexpr std::size_t PMR = 4;  // 打包微内核行数
constexpr std::size_t PNR = 8;  // 打包微内核列数 (两个 256-bit 向量 = 8 double)

void packed_i0_block(std::size_t n, const double* __restrict A,
                     const double* __restrict B, double* __restrict C,
                     std::size_t block, std::size_t i0,
                     double* __restrict A_pack, double* __restrict B_pack) {
    const std::size_t i_end = std::min(i0 + block, n);
    const std::size_t MC = i_end - i0;

    for (std::size_t k0 = 0; k0 < n; k0 += block) {
        const std::size_t k_end = std::min(k0 + block, n);
        const std::size_t KC = k_end - k0;

        // pack A 面板: A_pack[k*MC + r] = A[(i0+r)*n + (k0+k)]
        for (std::size_t k = 0; k < KC; ++k) {
            const double* src = A + i0 * n + (k0 + k);
            double* dst = A_pack + k * MC;
            for (std::size_t r = 0; r < MC; ++r) {
                dst[r] = src[r * n];
            }
        }

        for (std::size_t j0 = 0; j0 < n; j0 += block) {
            const std::size_t j_end = std::min(j0 + block, n);
            const std::size_t NC = j_end - j0;

            // pack B 面板: B_pack[k*NC + c] = B[(k0+k)*n + (j0+c)]
            for (std::size_t k = 0; k < KC; ++k) {
                std::memcpy(B_pack + k * NC, B + (k0 + k) * n + j0,
                            NC * sizeof(double));
            }

#if GEMM_HAS_AVX2_FMA
            const std::size_t ibulk = MC - (MC % PMR);
            const std::size_t jbulk = NC - (NC % PNR);

            // 主体微内核 [0, ibulk) x [0, jbulk)
            for (std::size_t i = 0; i < ibulk; i += PMR) {
                for (std::size_t j = 0; j < jbulk; j += PNR) {
                    __m256d acc[PMR][2];
                    for (std::size_t r = 0; r < PMR; ++r) {
                        acc[r][0] =
                            _mm256_loadu_pd(&C[(i0 + i + r) * n + (j0 + j)]);
                        acc[r][1] =
                            _mm256_loadu_pd(&C[(i0 + i + r) * n + (j0 + j) + 4]);
                    }
                    for (std::size_t k = 0; k < KC; ++k) {
                        const __m256d b0 = _mm256_loadu_pd(B_pack + k * NC + j);
                        const __m256d b1 =
                            _mm256_loadu_pd(B_pack + k * NC + j + 4);
                        for (std::size_t r = 0; r < PMR; ++r) {
                            const __m256d a =
                                _mm256_broadcast_sd(A_pack + k * MC + i + r);
                            acc[r][0] = _mm256_fmadd_pd(a, b0, acc[r][0]);
                            acc[r][1] = _mm256_fmadd_pd(a, b1, acc[r][1]);
                        }
                    }
                    for (std::size_t r = 0; r < PMR; ++r) {
                        _mm256_storeu_pd(&C[(i0 + i + r) * n + (j0 + j)],
                                         acc[r][0]);
                        _mm256_storeu_pd(&C[(i0 + i + r) * n + (j0 + j) + 4],
                                         acc[r][1]);
                    }
                }
            }
#else
            const std::size_t ibulk = 0;
            const std::size_t jbulk = 0;
#endif

            // 余数: 剩下的行 [ibulk, MC) 全部列
            for (std::size_t i = ibulk; i < MC; ++i) {
                double* Ci = C + (i0 + i) * n;
                const double* Ai = A + (i0 + i) * n;
                for (std::size_t k = 0; k < KC; ++k) {
                    const double a = Ai[k0 + k];
                    const double* Bk = B + (k0 + k) * n;
                    for (std::size_t j = 0; j < NC; ++j) {
                        Ci[j0 + j] += a * Bk[j0 + j];
                    }
                }
            }
            // 余数: 主体行 [0, ibulk) 剩下的列 [jbulk, NC)
            for (std::size_t i = 0; i < ibulk; ++i) {
                double* Ci = C + (i0 + i) * n;
                const double* Ai = A + (i0 + i) * n;
                for (std::size_t k = 0; k < KC; ++k) {
                    const double a = Ai[k0 + k];
                    const double* Bk = B + (k0 + k) * n;
                    for (std::size_t j = jbulk; j < NC; ++j) {
                        Ci[j0 + j] += a * Bk[j0 + j];
                    }
                }
            }
        }
    }
}

void gemm_packed(std::size_t n, const double* __restrict A,
                 const double* __restrict B, double* __restrict C,
                 std::size_t block) {
    if (block == 0 || block > n) block = n;
    std::vector<double> A_pack(block * block);
    std::vector<double> B_pack(block * block);
    for (std::size_t i0 = 0; i0 < n; i0 += block) {
        packed_i0_block(n, A, B, C, block, i0, A_pack.data(), B_pack.data());
    }
}

// ---------- Phase 7: Phase 6 + OpenMP 多线程 ----------
void gemm_packed_omp(std::size_t n, const double* __restrict A,
                     const double* __restrict B, double* __restrict C,
                     std::size_t block) {
    if (block == 0 || block > n) block = n;
#if defined(_OPENMP)
#pragma omp parallel
    {
        std::vector<double> A_pack(block * block);
        std::vector<double> B_pack(block * block);
#pragma omp for schedule(static)
        for (std::size_t i0 = 0; i0 < n; i0 += block) {
            packed_i0_block(n, A, B, C, block, i0, A_pack.data(), B_pack.data());
        }
    }
#else
    gemm_packed(n, A, B, C, block);
#endif
}

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
    std::setvbuf(stdout, nullptr, _IONBF, 0);

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
    std::printf(" 向量化: AVX2 + FMA  (Phase5: MR=%zu x NR=%zu | Phase6/7: MR=%zu x NR=%zu)\n",
                MR, NR, PMR, PNR);
#endif
    std::printf(" 分块大小 block = %zu\n", BLOCK);
#if defined(_OPENMP)
    std::printf(" OpenMP: 可用 (最大线程 %d)\n", omp_get_max_threads());
#else
    std::printf(" OpenMP: 不可用 (未链接 -fopenmp)\n");
#endif
    std::printf("============================================================\n");

    double* A = alloc_aligned64(N * N);
    double* B = alloc_aligned64(N * N);
    double* C_ref = alloc_aligned64(N * N);   // i-k-j 参考结果 (兼作 Phase 3)
    double* C_work = alloc_aligned64(N * N);  // 各优化阶段复用
    if (!A || !B || !C_ref || !C_work) {
        std::printf("内存分配失败\n");
        free_aligned64(A);
        free_aligned64(B);
        free_aligned64(C_ref);
        free_aligned64(C_work);
        return 1;
    }

    init_matrices(N, A, B);

    std::printf("64 字节对齐校验: A=%s B=%s C_ref=%s C_work=%s\n",
                is_aligned64(A) ? "OK" : "FAIL",
                is_aligned64(B) ? "OK" : "FAIL",
                is_aligned64(C_ref) ? "OK" : "FAIL",
                is_aligned64(C_work) ? "OK" : "FAIL");

    // Phase 3: 生成参考结果并计时
    std::fill(C_ref, C_ref + N * N, 0.0);
    double ms3 = 0.0;
    {
        const auto t0 = std::chrono::steady_clock::now();
        gemm_ikj(N, A, B, C_ref);
        const auto t1 = std::chrono::steady_clock::now();
        ms3 = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("[Phase 3] i-k-j (未分块)             : %9.1f ms  (%.3f GFLOPS)\n",
                    ms3, gflops_of(N, ms3));
    }

    // 通用: 计时 + 全矩阵校验 (写入 C_work, 与 C_ref 比较)
    auto bench = [&](const char* label, auto&& kernel) {
        std::fill(C_work, C_work + N * N, 0.0);
        const auto t0 = std::chrono::steady_clock::now();
        kernel();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("%s %9.1f ms  (%.3f GFLOPS)\n", label, ms, gflops_of(N, ms));
        const double e = max_rel_err(N, C_work, C_ref);
        std::printf("    全矩阵校验 vs i-k-j: max 相对误差 = %.3e  %s\n", e,
                    (e < 1e-9) ? "OK" : "FAIL");
        return ms;
    };

    double ms4 = bench("[Phase 4] i-k-j + Blocking          :",
                       [&] { gemm_tiled(N, A, B, C_work, BLOCK); });
    double ms5 = 0.0, ms6 = 0.0, ms7 = 0.0;
#if GEMM_HAS_AVX2_FMA
    ms5 = bench("[Phase 5] + AVX2 微内核 (8x4)       :",
                [&] { gemm_tiled_avx2(N, A, B, C_work, BLOCK); });
#endif
    ms6 = bench("[Phase 6] + Packing 微内核 (4x8)     :",
                [&] { gemm_packed(N, A, B, C_work, BLOCK); });
    ms7 = bench("[Phase 7] + OpenMP 多线程            :",
                [&] { gemm_packed_omp(N, A, B, C_work, BLOCK); });

    // 抽样校验 (独立点积)
    {
        std::mt19937_64 rng(20260825ULL);
        std::uniform_int_distribution<std::size_t> dist(0, N - 1);
        double werr = 0.0;
        for (int s = 0; s < 16; ++s) {
            const std::size_t r = dist(rng), c = dist(rng);
            const double ref = dot_ref(N, A, B, r, c);
            const double got = C_work[r * N + c];
            const double denom = std::fabs(ref) > 1e-30 ? std::fabs(ref) : 1e-30;
            werr = std::max(werr, std::fabs(got - ref) / denom);
        }
        std::printf("抽样校验 (16 个随机元素 vs 独立点积): max 相对误差 = %.3e  %s\n",
                    werr, (werr < 1e-9) ? "OK" : "FAIL");
    }

    // 小矩阵补测 Phase 1 朴素 baseline
    if (N <= 1024) {
        bench("[Phase 1] i-j-k (朴素 baseline)      :",
              [&] { gemm_naive(N, A, B, C_work); });
    } else {
        std::printf("[Phase 1] i-j-k (朴素 baseline)      : 跳过 (N > 1024, 过于缓慢)\n");
    }

    // --tune: 扫描 Phase 6 的 block 大小
    if (tune) {
        const std::size_t candidates[] = {32, 48, 64, 96, 128, 160, 192, 256};
        std::size_t best = BLOCK;
        double best_ms = ms6;
        std::printf("------------------------------------------------------------\n");
        std::printf("Block 调优 (Phase 6, Packing):\n");
        for (std::size_t b : candidates) {
            if (b > N) continue;
            std::fill(C_work, C_work + N * N, 0.0);
            const auto t0 = std::chrono::steady_clock::now();
            gemm_packed(N, A, B, C_work, b);
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

    std::printf("------------------------------------------------------------\n");
    if (ms3 > 0.0) {
        if (ms4 > 0.0) std::printf("Phase3 -> Phase4 加速比: %.2fx\n", ms3 / ms4);
        if (ms5 > 0.0) std::printf("Phase3 -> Phase5 加速比: %.2fx\n", ms3 / ms5);
        if (ms6 > 0.0) std::printf("Phase3 -> Phase6 加速比: %.2fx\n", ms3 / ms6);
        if (ms7 > 0.0) std::printf("Phase3 -> Phase7 加速比: %.2fx\n", ms3 / ms7);
    }

    free_aligned64(A);
    free_aligned64(B);
    free_aligned64(C_ref);
    free_aligned64(C_work);
    return 0;
}










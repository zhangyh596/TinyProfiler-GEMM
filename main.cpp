// GEMM Phase 3: i-k-j + 64B 对齐 + -march=native / -ffast-math 向量化。
// 构建: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
//
// 注: Windows/MinGW 无 std::aligned_alloc 与 posix_memalign,
//     故采用经典手动对齐: malloc 多分配, 原始指针存于对齐地址之前, free 释放原始指针。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>  // std::malloc, std::free

namespace {
constexpr std::size_t ALIGN = 64;  // 对齐到 64 字节 (CPU Cache Line)

// 分配 n 个 double, 返回 64 字节对齐的指针 (失败返回 nullptr)
double* alloc_aligned64(std::size_t n) {
    const std::size_t bytes = n * sizeof(double);
    void* raw = std::malloc(bytes + ALIGN - 1 + sizeof(void*));
    if (!raw) return nullptr;

    // 向上对齐, 并在对齐地址之前 sizeof(void*) 字节处保存原始指针
    const std::uintptr_t aligned =
        (reinterpret_cast<std::uintptr_t>(raw) + sizeof(void*) + ALIGN - 1) &
        ~static_cast<std::uintptr_t>(ALIGN - 1);
    reinterpret_cast<void**>(aligned)[-1] = raw;
    return reinterpret_cast<double*>(aligned);
}

// 释放 alloc_aligned64 返回的指针
void free_aligned64(double* p) {
    if (p) std::free(reinterpret_cast<void**>(p)[-1]);
}
}  // namespace

int main() {
    constexpr std::size_t N = 1024;  // 方阵维度 (M = K = N)

    double* A = alloc_aligned64(N * N);
    double* B = alloc_aligned64(N * N);
    double* C = alloc_aligned64(N * N);
    if (!A || !B || !C) {
        std::printf("内存分配失败\n");
        return 1;
    }

    // 初始化 A、B，清零 C
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t k = 0; k < N; ++k) {
            A[i * N + k] = static_cast<double>((i * N + k) % 100) * 0.01;
            B[i * N + k] = static_cast<double>((k * N + i) % 100) * 0.01;
        }
    }
    std::fill(C, C + N * N, 0.0);

    // 打印对齐后的首地址, 验证是否可被 64 整除
    auto is_aligned64 = [](const double* p) {
        return reinterpret_cast<std::uintptr_t>(p) % ALIGN == 0;
    };
    std::printf("首地址: A=%p B=%p C=%p\n", static_cast<const void*>(A),
                static_cast<const void*>(B), static_cast<const void*>(C));
    std::printf("64 字节对齐校验: A=%s B=%s C=%s\n",
                is_aligned64(A) ? "OK" : "FAIL",
                is_aligned64(B) ? "OK" : "FAIL",
                is_aligned64(C) ? "OK" : "FAIL");

    const auto t0 = std::chrono::steady_clock::now();

    // i-k-j 三重循环: B 按行连续访问, C 的第 i 行常驻 cache
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t k = 0; k < N; ++k) {
            const double a = A[i * N + k];
            for (std::size_t j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 正确性校验: 重算 C[0][0]。
    // 注: -ffast-math 允许重排序/融合运算, 结果可能与参考值差几个 ULP,
    //     故改用相对误差判断而非精确 ==。
    double ref = 0.0;
    for (std::size_t k = 0; k < N; ++k) {
        ref += A[0 * N + k] * B[k * N + 0];
    }
    const double rel_err = std::fabs(C[0] - ref) / std::fabs(ref);

    const double gflops = 2.0 * N * N * N / (ms * 1e-3) / 1e9;
    std::printf("N = %zu\n", N);
    std::printf("GEMM (i-k-j) 耗时: %.3f ms  (%.3f GFLOPS)\n", ms, gflops);
    std::printf("校验 C[0][0]: 计算值 = %f, 参考值 = %f, 相对误差 = %.3e, %s\n",
                C[0], ref, rel_err, (rel_err < 1e-9) ? "OK" : "FAIL");

    // Linux perf stat 硬件计数器抓取提示
    std::printf("\n====================================================\n");
    std::printf("Linux perf stat 抓取硬件计数器 (需 Linux + perf 工具):\n");
    std::printf("  perf stat -e L1-dcache-load-misses,instructions,cycles ./build/gemm\n");
    std::printf("若提示权限不足 (Access to performance monitoring... 报错), 先执行:\n");
    std::printf("  sudo sysctl kernel.perf_event_paranoid=1\n");
    std::printf("或改用通用事件别名:\n");
    std::printf("  perf stat -e cache-misses,instructions,cycles ./build/gemm\n");
    std::printf("====================================================\n");

    free_aligned64(A);
    free_aligned64(B);
    free_aligned64(C);
    return 0;
}

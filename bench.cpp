#include <rknn_matmul_api.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <span>
#include <cstring>
#include <thread>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <csignal>

// ============================================================
// RK3588 NPU 3-Core Full Load Stress Test
// 
// 목적: 3개 NPU 코어를 동시에 full-load로 구동하여
//       최대 전력 소모 상태를 만들고, 배터리 소모 테스트에 활용
//
// 빌드: g++ npu_stress.cpp -o npu_stress -lrknnrt -lpthread -O3 -std=c++20
// 실행: taskset -c 4-7 ./npu_stress          (A76 코어에서 실행 권장)
//
// NPU 코어 스펙 (per core):
//   - 1GHz clock
//   - INT8: 1024 ops/cycle = ~1 TOPS/core → 3 TOPS total
//   - FP16: 512 ops/cycle  = ~0.5 TFLOPS/core → 1.5 TFLOPS total
//   - INT4: 2048 ops/cycle = ~2 TOPS/core → 6 TOPS total
// ============================================================

template <typename T, typename U>
void fill_random(std::span<T> data, U min, U max)
{
    using Dist = std::conditional_t<std::is_integral_v<T>,
        std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;
    std::random_device rd;
    std::mt19937 gen(rd());
    Dist dis(min, max);
    for (auto& x : data) x = dis(gen);
}

struct RKNNMatMul
{
    int m, k, n;
    rknn_matmul_type type;
    rknn_matmul_ctx ctx = 0;
    rknn_matmul_info info;
    rknn_matmul_io_attr attr;
    rknn_tensor_mem *A = nullptr, *B = nullptr, *C = nullptr;
    bool valid = false;

    RKNNMatMul(int m, int k, int n, rknn_matmul_type type,
               bool ac_native = true, bool b_native = true)
        : m(m), k(k), n(n), type(type)
    {
        memset(&info, 0, sizeof(info));
        info.M = m; info.K = k; info.N = n;
        info.type = type;
        info.B_layout = b_native;
        info.AC_layout = ac_native;

        memset(&attr, 0, sizeof(attr));
        int ret = rknn_matmul_create(&ctx, &info, &attr);
        if (ret != 0) {
            std::cerr << "rknn_matmul_create failed: " << ret << std::endl;
            return;
        }

        // Allocate and fill input matrices
        size_t a_bytes, b_bytes;
        if (type == RKNN_INT8_MM_INT8_TO_INT32) {
            a_bytes = m * k; b_bytes = k * n;
        } else if (type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32) {
            a_bytes = m * k * 2; b_bytes = k * n * 2;
        } else if (type == RKNN_INT4_MM_INT4_TO_INT16) {
            a_bytes = m * k / 2; b_bytes = k * n / 2;
        } else return;

        void* a_data = malloc(a_bytes);
        void* b_data = malloc(b_bytes);

        if (type == RKNN_INT8_MM_INT8_TO_INT32) {
            fill_random(std::span<int8_t>((int8_t*)a_data, m*k), -128, 127);
            fill_random(std::span<int8_t>((int8_t*)b_data, k*n), -128, 127);
        } else if (type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32) {
            fill_random(std::span<uint16_t>((uint16_t*)a_data, m*k), -1.0, 1.0);
            fill_random(std::span<uint16_t>((uint16_t*)b_data, k*n), -1.0, 1.0);
        } else {
            fill_random(std::span<int8_t>((int8_t*)a_data, m*k/2), -8, 7);
            fill_random(std::span<int8_t>((int8_t*)b_data, k*n/2), -8, 7);
        }

        A = rknn_create_mem(ctx, attr.A.size);
        B = rknn_create_mem(ctx, attr.B.size);
        C = rknn_create_mem(ctx, attr.C.size);
        if (!A || !B || !C) {
            std::cerr << "rknn_create_mem failed" << std::endl;
            free(a_data); free(b_data); return;
        }

        memcpy(A->virt_addr, a_data, A->size);
        memcpy(B->virt_addr, b_data, B->size);
        free(a_data); free(b_data);

        rknn_matmul_set_io_mem(ctx, A, &attr.A);
        rknn_matmul_set_io_mem(ctx, B, &attr.B);
        rknn_matmul_set_io_mem(ctx, C, &attr.C);
        valid = true;
    }

    int run() { return rknn_matmul_run(ctx); }

    ~RKNNMatMul() {
        if (A) rknn_destroy_mem(ctx, A);
        if (B) rknn_destroy_mem(ctx, B);
        if (C) rknn_destroy_mem(ctx, C);
        if (ctx) rknn_matmul_destroy(ctx);
    }
};

// ============================================================
// Per-core stress worker
// ============================================================
struct CoreStats {
    std::atomic<uint64_t> total_runs{0};
    std::atomic<uint64_t> total_ns{0};
    std::atomic<double>   peak_gops{0};
};

void stress_worker(int core_id, int m, int k, int n,
                   rknn_matmul_type type,
                   std::atomic<bool>& running,
                   CoreStats& stats)
{
    // native layout = 최대 성능 (SRAM 최적화된 데이터 배치)
    RKNNMatMul matmul(m, k, n, type, /*ac_native=*/true, /*b_native=*/true);
    if (!matmul.valid) {
        std::cerr << "[Core " << core_id << "] Init failed!" << std::endl;
        return;
    }

    std::cout << "[Core " << core_id << "] Ready: "
              << m << "x" << k << "x" << n << std::endl;

    // Warm-up: 5 runs
    for (int i = 0; i < 5; i++) matmul.run();

    uint64_t ops_per_run = (uint64_t)m * n * (2ULL * k - 1);

    while (running.load()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul.run();
        auto t1 = std::chrono::high_resolution_clock::now();

        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        double gops = (double)ops_per_run / (double)ns; // GOPS

        stats.total_runs.fetch_add(1);
        stats.total_ns.fetch_add(ns);

        // Update peak (relaxed is fine for monitoring)
        double cur_peak = stats.peak_gops.load();
        while (gops > cur_peak &&
               !stats.peak_gops.compare_exchange_weak(cur_peak, gops));
    }

    std::cout << "[Core " << core_id << "] Stopped." << std::endl;
}

// ============================================================
// Monitor thread: 1초마다 상태 출력
// ============================================================
void monitor_thread(std::atomic<bool>& running,
                    CoreStats stats[3],
                    rknn_matmul_type type)
{
    const char* type_str;
    double theoretical_per_core;

    switch(type) {
    case RKNN_INT8_MM_INT8_TO_INT32:
        type_str = "INT8";
        theoretical_per_core = 1000.0; // 1 TOPS = 1000 GOPS
        break;
    case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
        type_str = "FP16";
        theoretical_per_core = 500.0;  // 0.5 TFLOPS
        break;
    case RKNN_INT4_MM_INT4_TO_INT16:
        type_str = "INT4";
        theoretical_per_core = 2000.0; // 2 TOPS
        break;
    default:
        type_str = "???";
        theoretical_per_core = 1000.0;
    }

    double theoretical_total = theoretical_per_core * 3;

    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════╗\n"
              << "║  RK3588 NPU 3-Core Stress Test (" << type_str << ")                       ║\n"
              << "║  Theoretical max: " << std::fixed << std::setprecision(1)
              << theoretical_total << " GOPS (" << type_str << ")                           ║\n"
              << "║  Press Ctrl+C to stop                                       ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n"
              << std::endl;

    uint64_t prev_runs[3] = {0, 0, 0};
    int sec = 0;

    while (running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        sec++;

        double core_gops[3], total_gops = 0;

        std::cout << "── [" << sec << "s] ─────────────────────────────\n";
        for (int i = 0; i < 3; i++) {
            uint64_t runs = stats[i].total_runs.load();
            uint64_t ns = stats[i].total_ns.load();
            uint64_t delta_runs = runs - prev_runs[i];
            prev_runs[i] = runs;

            core_gops[i] = (runs > 0) ? stats[i].peak_gops.load() : 0;
            total_gops += core_gops[i];

            double util = (core_gops[i] / theoretical_per_core) * 100.0;

            std::cout << "  Core " << i
                      << ": " << std::setw(7) << std::fixed << std::setprecision(1)
                      << core_gops[i] << " GOPS"
                      << "  (" << std::setprecision(1) << util << "% efficiency)"
                      << "  runs/s: " << delta_runs
                      << "\n";
        }

        double total_util = (total_gops / theoretical_total) * 100.0;
        std::cout << "  TOTAL : " << std::setw(7) << std::fixed << std::setprecision(1)
                  << total_gops << " GOPS"
                  << "  (" << std::setprecision(1) << total_util << "% of "
                  << theoretical_total << " GOPS theoretical)\n"
                  << std::endl;
    }
}

// ============================================================
// Main
// ============================================================
std::atomic<bool> g_running{true};

void signal_handler(int) { g_running.store(false); }

int main(int argc, char* argv[])
{
    signal(SIGINT, signal_handler);

    // --------------------------------------------------------
    // 최적 행렬 크기 선택 가이드:
    //
    // NPU utilization을 최대화하려면:
    //   1) M이 충분히 커야 함 (≥256). M=1은 GEMV로 효율 급락
    //   2) K, N은 4096 이상이면 compute-bound 영역 진입
    //   3) native layout 사용 필수
    //   4) alignment 준수: INT8→32byte, FP16→16byte, INT4→64byte
    //
    // 추천 설정 (compute-bound, 최대 MAC 활용):
    //   INT8:  M=1024, K=4096, N=4096  → ~34B ops/run
    //   FP16:  M=512,  K=4096, N=4096  → ~17B ops/run
    //   INT4:  M=1024, K=4096, N=4096  → ~34B ops/run
    //
    // M을 더 키우면 단일 run 시간이 길어지지만 효율은 비슷.
    // M이 너무 작으면 (1~8) memory-bound가 되어 MAC 활용률 급락.
    // --------------------------------------------------------

    int M = 1024;
    int K = 4096;
    int N = 4096;
    rknn_matmul_type type = RKNN_INT8_MM_INT8_TO_INT32;

    // Parse optional args
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        int t = std::atoi(argv[4]);
        if (t == 0) type = RKNN_INT8_MM_INT8_TO_INT32;
        else if (t == 1) type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        else if (t == 2) type = RKNN_INT4_MM_INT4_TO_INT16;
    }

    std::cout << "Matrix size: M=" << M << " K=" << K << " N=" << N << "\n";
    std::cout << "Ops per matmul: "
              << (uint64_t)M * N * (2ULL*K - 1) / 1e9 << " GOPS\n";

    // 3 코어 각각에 독립 matmul 인스턴스
    CoreStats stats[3];

    std::thread workers[3];
    for (int i = 0; i < 3; i++) {
        workers[i] = std::thread(stress_worker, i, M, K, N,
                                 type, std::ref(g_running), std::ref(stats[i]));
    }

    // Monitor
    std::thread mon(monitor_thread, std::ref(g_running), stats, type);

    // Wait for Ctrl+C
    for (auto& w : workers) w.join();
    mon.join();

    // Final summary
    std::cout << "\n═══ Final Summary ═══\n";
    for (int i = 0; i < 3; i++) {
        uint64_t runs = stats[i].total_runs.load();
        uint64_t ns = stats[i].total_ns.load();
        double avg_ms = (runs > 0) ? (double)ns / runs / 1e6 : 0;
        std::cout << "Core " << i
                  << ": " << runs << " runs"
                  << ", avg " << std::fixed << std::setprecision(2) << avg_ms << " ms/run"
                  << ", peak " << std::setprecision(1) << stats[i].peak_gops.load() << " GOPS\n";
    }

    return 0;
}
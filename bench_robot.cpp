#include <rknn_matmul_api.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <thread>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <csignal>

// ============================================================
// RK3588 NPU 3-Core Full Load Stress Test
//
// 빌드: g++ npu_stress.cpp -o npu_stress -I/usr/include/rknn -lrknnrt -lpthread -O3 -std=c++17
// 실행: taskset -c 4-7 ./npu_stress [M K N type(0=INT8,1=FP16)]
//
// NPU 코어 스펙 (per core):
//   INT8:  ~1 TOPS/core  → 3 TOPS total
//   FP16:  ~0.5 TFLOPS   → 1.5 TFLOPS total
// ============================================================

// -------- C++17 호환 fill_random (std::span 제거) --------
template <typename T>
void fill_random(T* data, size_t count, T min_val, T max_val)
{
    using Dist = typename std::conditional<std::is_integral<T>::value,
        std::uniform_int_distribution<T>,
        std::uniform_real_distribution<T>>::type;
    std::random_device rd;
    std::mt19937 gen(rd());
    Dist dis(min_val, max_val);
    for (size_t i = 0; i < count; ++i)
        data[i] = dis(gen);
}

// -------- NPU 코어 마스크 배열 (Core 0, 1, 2) --------
static const rknn_core_mask CORE_MASKS[3] = {
    RKNN_NPU_CORE_0,
    RKNN_NPU_CORE_1,
    RKNN_NPU_CORE_2,
};

// ============================================================
// RKNNMatMul wrapper
// ============================================================
struct RKNNMatMul
{
    int m, k, n;
    rknn_tensor_type type;      // 실제 헤더: rknn_tensor_type
    rknn_matmul_ctx  ctx = 0;
    rknn_matmul_info info;
    rknn_matmul_io_attr attr;
    rknn_tensor_mem *A = nullptr, *B = nullptr, *C = nullptr;
    bool valid = false;

    // native_layout : B 행렬 native layout (0=normal, 1=native)
    // perf_layout   : A/C 행렬 perf layout  (0=normal, 1=perf)
    // core_mask     : 이 인스턴스를 실행할 NPU 코어
    RKNNMatMul(int m, int k, int n, rknn_tensor_type type,
               int native_layout, int perf_layout,
               rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO)
        : m(m), k(k), n(n), type(type)
    {
        memset(&info, 0, sizeof(info));
        info.M             = m;
        info.K             = k;
        info.N             = n;
        info.type          = type;
        info.native_layout = native_layout;
        info.perf_layout   = perf_layout;

        memset(&attr, 0, sizeof(attr));
        int ret = rknn_matmul_create(&ctx, &info, &attr);
        if (ret != 0) {
            std::cerr << "rknn_matmul_create failed: " << ret << std::endl;
            return;
        }

        // 코어 고정
        ret = rknn_matmul_set_core_mask(ctx, core_mask);
        if (ret != 0) {
            std::cerr << "rknn_matmul_set_core_mask failed: " << ret << std::endl;
            return;
        }

        // 입력 데이터 할당 및 초기화
        size_t a_bytes, b_bytes;
        if (type == RKNN_TENSOR_INT8) {
            a_bytes = (size_t)m * k;
            b_bytes = (size_t)k * n;
        } else if (type == RKNN_TENSOR_FLOAT16) {
            a_bytes = (size_t)m * k * 2;
            b_bytes = (size_t)k * n * 2;
        } else {
            std::cerr << "Unsupported type" << std::endl;
            return;
        }

        void* a_data = malloc(a_bytes);
        void* b_data = malloc(b_bytes);

        if (type == RKNN_TENSOR_INT8) {
            fill_random(reinterpret_cast<int8_t*>(a_data), (size_t)m * k, (int8_t)-128, (int8_t)127);
            fill_random(reinterpret_cast<int8_t*>(b_data), (size_t)k * n, (int8_t)-128, (int8_t)127);
        } else {
            fill_random(reinterpret_cast<uint16_t*>(a_data), (size_t)m * k, (uint16_t)0, (uint16_t)65535);
            fill_random(reinterpret_cast<uint16_t*>(b_data), (size_t)k * n, (uint16_t)0, (uint16_t)65535);
        }

        A = rknn_create_mem(ctx, attr.A.size);
        B = rknn_create_mem(ctx, attr.B.size);
        C = rknn_create_mem(ctx, attr.C.size);
        if (!A || !B || !C) {
            std::cerr << "rknn_create_mem failed" << std::endl;
            free(a_data); free(b_data);
            return;
        }

        memcpy(A->virt_addr, a_data, A->size);
        memcpy(B->virt_addr, b_data, B->size);
        free(a_data);
        free(b_data);

        rknn_matmul_set_io_mem(ctx, A, &attr.A);
        rknn_matmul_set_io_mem(ctx, B, &attr.B);
        rknn_matmul_set_io_mem(ctx, C, &attr.C);
        valid = true;
    }

    int run() { return rknn_matmul_run(ctx); }

    ~RKNNMatMul()
    {
        if (A) rknn_destroy_mem(ctx, A);
        if (B) rknn_destroy_mem(ctx, B);
        if (C) rknn_destroy_mem(ctx, C);
        if (ctx) rknn_matmul_destroy(ctx);
    }
};

// ============================================================
// Per-core stats
// ============================================================
struct CoreStats {
    std::atomic<uint64_t> total_runs{0};
    std::atomic<uint64_t> total_ns{0};
    std::atomic<double>   peak_gops{0.0};
};

// ============================================================
// Stress worker (코어 1개 담당)
// ============================================================
void stress_worker(int core_id, int m, int k, int n,
                   rknn_tensor_type type,
                   std::atomic<bool>& running,
                   CoreStats& stats)
{
    // native_layout=1, perf_layout=1 → 최대 성능
    RKNNMatMul matmul(m, k, n, type, 1, 1, CORE_MASKS[core_id]);
    if (!matmul.valid) {
        std::cerr << "[Core " << core_id << "] Init failed!" << std::endl;
        return;
    }

    std::cout << "[Core " << core_id << "] Ready: "
              << m << "x" << k << "x" << n << std::endl;

    // Warm-up
    for (int i = 0; i < 5; i++) matmul.run();

    const uint64_t ops_per_run = (uint64_t)m * n * (2ULL * k - 1);

    while (running.load()) {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmul.run();
        auto t1 = std::chrono::high_resolution_clock::now();

        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        double gops = (double)ops_per_run / static_cast<double>(ns); // GOPS

        stats.total_runs.fetch_add(1);
        stats.total_ns.fetch_add(ns);

        // peak 갱신 (compare_exchange_weak: C++11~)
        double cur = stats.peak_gops.load();
        while (gops > cur && !stats.peak_gops.compare_exchange_weak(cur, gops)) {}
    }

    std::cout << "[Core " << core_id << "] Stopped." << std::endl;
}

// ============================================================
// Monitor thread: 1초마다 상태 출력
// ============================================================
void monitor_thread(std::atomic<bool>& running,
                    CoreStats* stats,          // CoreStats[3]
                    rknn_tensor_type type)
{
    const char* type_str;
    double theoretical_per_core;

    if (type == RKNN_TENSOR_INT8) {
        type_str = "INT8";
        theoretical_per_core = 1000.0; // 1 TOPS = 1000 GOPS
    } else {
        type_str = "FP16";
        theoretical_per_core = 500.0;  // 0.5 TFLOPS
    }
    const double theoretical_total = theoretical_per_core * 3;

    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════╗\n"
              << "║  RK3588 NPU 3-Core Stress Test (" << type_str << ")                    ║\n"
              << "║  Theoretical max: " << std::fixed << std::setprecision(1)
              << theoretical_total << " GOPS (" << type_str << ")                     ║\n"
              << "║  Press Ctrl+C to stop                                        ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n"
              << std::endl;

    uint64_t prev_runs[3] = {0, 0, 0};
    int sec = 0;

    while (running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        sec++;

        double total_gops = 0;
        std::cout << "── [" << sec << "s] ─────────────────────────────\n";

        for (int i = 0; i < 3; i++) {
            uint64_t runs = stats[i].total_runs.load();
            uint64_t delta = runs - prev_runs[i];
            prev_runs[i] = runs;

            double gops = stats[i].peak_gops.load();
            total_gops += gops;
            double util = gops / theoretical_per_core * 100.0;

            std::cout << "  Core " << i
                      << ": " << std::setw(7) << std::fixed << std::setprecision(1)
                      << gops << " GOPS"
                      << "  (" << std::setprecision(1) << util << "% efficiency)"
                      << "  runs/s: " << delta << "\n";
        }

        double total_util = total_gops / theoretical_total * 100.0;
        std::cout << "  TOTAL : " << std::setw(7) << std::fixed << std::setprecision(1)
                  << total_gops << " GOPS"
                  << "  (" << std::setprecision(1) << total_util << "% of "
                  << theoretical_total << " GOPS theoretical)\n\n";
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

    int M = 1024, K = 4096, N = 4096;
    rknn_tensor_type type = RKNN_TENSOR_INT8;

    if (argc >= 4) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        int t = std::atoi(argv[4]);
        type = (t == 1) ? RKNN_TENSOR_FLOAT16 : RKNN_TENSOR_INT8;
    }

    std::cout << "Matrix: M=" << M << " K=" << K << " N=" << N << "\n";
    std::cout << "Ops/matmul: "
              << (double)((uint64_t)M * N * (2ULL*K - 1)) / 1e9 << " GOPS\n";

    CoreStats stats[3];

    // 3개 worker 스레드 (각각 NPU Core 0/1/2에 고정)
    std::thread workers[3];
    for (int i = 0; i < 3; i++) {
        workers[i] = std::thread(stress_worker, i, M, K, N,
                                 type, std::ref(g_running), std::ref(stats[i]));
    }

    std::thread mon(monitor_thread, std::ref(g_running), stats, type);

    for (auto& w : workers) w.join();
    mon.join();

    // Final summary
    std::cout << "\n═══ Final Summary ═══\n";
    for (int i = 0; i < 3; i++) {
        uint64_t runs = stats[i].total_runs.load();
        uint64_t ns   = stats[i].total_ns.load();
        double avg_ms = (runs > 0) ? (double)ns / runs / 1e6 : 0.0;
        std::cout << "Core " << i
                  << ": " << runs << " runs"
                  << ", avg " << std::fixed << std::setprecision(2) << avg_ms << " ms/run"
                  << ", peak " << std::setprecision(1) << stats[i].peak_gops.load() << " GOPS\n";
    }

    return 0;
}
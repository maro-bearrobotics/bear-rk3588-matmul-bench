# RK3588 matmul benchmark

Quick and dirty benchmarking tool to measure the performance of RK3588 NPU. 

## How to use

```
g++ bench.cpp -o bench -lrknnrt -lpthread -O3 -std=c++20
taskset -c 4-7 ./bench 1024 4096 4096 0
```

## How to use in robot (C++ 17)
```
g++ bench_robot.cpp -o bench -I/usr/include/rknn -lrknnrt -lpthread -O3 -std=c++17

g++ bench_robot.cpp -o bench -lrknnrt -lpthread -O3 -std=c++17

sudo ./run_stress_test.sh cpu 60     # CPU-only baseline
sudo ./run_stress_test.sh npu 60     # NPU-only baseline
sudo ./run_stress_test.sh both 60    # Combined interference test

```

To see NPU Utils  and temperature (different terminal)
- `sudo watch  -n 1 'echo "NPU temp: $(( $(cat /sys/class/thermal/thermal_zone6/temp) / 1000 ))C"; echo "NPU load: $(cat /sys/kernel/debug/rknpu/load 2>/dev/null || echo N/A)"'`



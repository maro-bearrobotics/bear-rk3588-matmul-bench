# RK3588 matmul benchmark

Quick and dirty benchmarking tool to measure the performance of RK3588 NPU. 

## How to use

```
g++ bench.cpp -o bench -lrknnrt -lpthread -O3 -std=c++20
taskset -c 4-7 ./bench 1024 4096 4096 0
```

To see NPU Utils (different terminal)
- `sudo watch -n 1 cat /sys/kernel/debug/rknpu/load`
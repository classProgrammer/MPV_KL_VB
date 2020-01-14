# CUDA Terms
- Grid: Consists of Blocks
- Block: Consists of e.g. 1024 Threads
- Thread: One e.g. Function call on the Device
- Warp: Block of 32 Threads

# Answers
- GPUs try to hide latencies by having “a lot of work in flight”

# GPGPU
- General Purpose Graphic Processing Unit
- Fine Grained SIMD Architecture

## CPU VS GPU
![](./img/cpuvsgpu.PNG)

## Nvidia SM Architecture
![](./img/smarch.PNG)

## CUDA Memory Architecture
![](./img/cmem.PNG)

## Storage of Values
![](./img/store1.PNG)
![](./img/store2.PNG)
![](./img/bsmem.PNG)

## Tuning Tipps
![](./img/tuning.PNG)
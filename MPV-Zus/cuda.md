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
![](cpuvsgpu.png)

## Nvidia SM Architecture
![](smarch.png)

## CUDA Memory Architecture
![](cmem.png)

## Storage of Values
![](store1.png)
![](store2.png)
![](bsmem.png)

## Tuning Tipps
![](tuning.png)
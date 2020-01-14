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
![](cpuvsgpu.PNG)

## Nvidia SM Architecture
![](smarch.PNG)

## CUDA Memory Architecture
![](cmem.PNG)

## Storage of Values
![](store1.PNG)
![](store2.PNG)
![](bsmem.PNG)

## Tuning Tipps
![](tuning.PNG)
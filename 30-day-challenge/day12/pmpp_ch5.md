5.1. Consider the matrix addition in Exercise 3.1.
Can one use shared memory to reduce the global memory bandwidth consumption? Hint: analyze the elements accessed by each thread and see if there is any commonality between threads.
Ans: Day 11


5.2. Draw the equivalent of Figure 5.6 for a 8×8 matrix multiplication with 2×2 tiling and 4×4 tiling.
Verify that the reduction in global memory bandwidth is indeed proportional to the dimension size of the tiles.


5.3. What type of incorrect execution behavior can happen if one forgots to use syncthreads() in the kernel of Figure 5.12?
*If `__syncthreads()` is missing, some threads may try to read data from shared memory before other threads have finished writing to it. This can cause race conditions, leading to incorrect and unpredictable results.*



5.4. Assuming capacity was not an issue for registers or shared memory,
give one case that it would be valuable to use shared memory instead of registers to hold values fetched from global memory? Explain your answer.
*A valuable case for using shared memory instead of registers is in matrix multiplication. In a naive implementation, each element of the input matrices is read multiple times from global memory, leading to \( O(N^3) \) memory accesses. By using shared memory to store small **tiles** of the matrices, each element is loaded only once per tile, reducing the total memory accesses to \( O(N^2) \). This significantly improves performance by reducing global memory bandwidth usage.*

5.5. For our tiled matrix–matrix multiplication kernel, if we use a 32×32 tile, what is the reduction of memory bandwidth usage for input matrices M and N?
a. 1/8 of the original usage
b. 1/16 of the original usage
c. 1/32 of the original usage
d. 1/64 of the original usage

Ans:
### **Before Using Tiles (Naïve Approach)**
- Every thread loads numbers from **global memory** each time it needs them.
- If we are multiplying two big matrices, we end up reading the same numbers **again and again**.
- This wastes a lot of **memory bandwidth** because fetching from global memory is slow.

### **After Using Tiles (Optimized with Shared Memory)**
- Instead of every thread loading the same numbers multiple times, we use a **shared memory tile** (a small cache).
- A **32×32 tile** means that **each value is loaded once per tile** and then reused **32 times** by different threads.
- This **reduces the number of times we need to fetch data from global memory**.

### **How Much Less Memory is Used?**
- Since each value is now **reused 32 times**, the total memory bandwidth usage drops to **1/32** of what it was before.

So the correct answer is:
**c. 1/32 of the original usage.**


5.6. Assume that a kernel is launched with 1,000 thread blocks each of which has 512 threads.
If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?
a. 1
b. 1,000
c. 512
d. 512,000

Ans: 1000 * 512;


5.7. In the previous question, if a variable is declared as a shared memory variable,
how many versions of the variable will be created through the lifetime of the execution of the kernel?
a. 1
b. 1,000
c. 512
d. 51,200
Ans: 1000, one variable for each block


5.8. Explain the difference between shared memory and L1 cache.
### **Difference Between Shared Memory and L1 Cache in CUDA**

| Feature          | **Shared Memory** | **L1 Cache** |
|-----------------|------------------|-------------|
| **Scope**        | Per **thread block** (shared among all threads in a block) | Per **Streaming Multiprocessor (SM)** (shared across multiple thread blocks) |
| **Control**      | **Explicitly managed** by the programmer | **Implicitly managed** by hardware |
| **Usage**        | Used for **storing frequently accessed data** within a block to reduce global memory accesses | Used for **caching global memory** to reduce latency automatically |
| **Latency**      | Very low (similar to register access) | Slightly higher than shared memory but much lower than global memory |
| **Size**         | Usually **up to 48 KB per block** (configurable with L1) | Typically **up to 48 KB per SM** (split between L1 and shared memory) |
| **Access Pattern** | Optimized for **known access patterns** where threads in a block cooperate | Works well for **unknown or irregular memory access patterns** |
| **Bank Conflicts** | Possible due to memory **banking** (can slow down access if not optimized) | No bank conflicts, but cache **evictions** can occur |

### **Key Takeaways:**
- **Shared memory is like a manually managed cache**, useful when you know how data should be reused in a block.
- **L1 cache is automatically managed by hardware**, helping with unpredictable memory access patterns.
- **Shared memory is faster than L1 cache**, but requires careful handling to avoid bank conflicts.



5.9. Consider performing a matrix multiplication of two input matrices with dimensions N×N. How many times is each element in the input matrices requested from global memory when:
a. There is no tiling?
b. Tiles of size T×T are used?

Your answer is close, but let's refine it for clarity and correctness.

---

### **(a) No Tiling (Naïve Matrix Multiplication)**
- Each element in the output matrix \( C(i, j) \) is computed as:
  \[
  C(i, j) = \sum_{k} M(i, k) \times N(k, j)
  \]
- Each thread computes **one element** of \( C \), requiring:
  - **One full row** of \( M \) (i.e., \( N \) elements)
  - **One full column** of \( N \) (i.e., \( N \) elements)
- Since there are **\( N^2 \) elements in \( C \)**, and each computation accesses an entire **row of \( M \) and a column of \( N \)**:
  - **Total memory accesses = \( N^3 \) (each element is loaded \( N \) times).**

✅ **Corrected Answer:**
\[
\text{Each element in the input matrices is requested } N \text{ times, leading to a total of } O(N^3) \text{ accesses}.
\]

---

### **(b) With Tiling (Tile Size = \( T \times T \))**
- Each **thread block** loads a **tile** of \( M \) and a **tile** of \( N \) into shared memory.
- Each tile is reused **T times** before a new tile is loaded from global memory.
- This reduces the number of memory accesses **per element**.

The total number of memory accesses **reduces from \( O(N^3) \) to \( O(N^2 / T) \)**.

✅ **Corrected Answer:**
$$
\[
\text{Each element is now requested only } O(N/T) \text{ times, leading to a total of } O(N^2) \text{ accesses}.
\]
$$


5.10. A kernel performs 36 floating-point operations and 7 32-bit word global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute- or memory-bound.
a. Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/s.
b. Peak FLOPS=300 GFLOPS, peak memory bandwidth=250 GB/s.

1. First, let's calculate the bytes accessed per thread:
   * 7 words × 4 bytes/word = 28 bytes per thread

2. Let's calculate the arithmetic intensity:
   * Arithmetic intensity = FLOPs / bytes accessed
   * Arithmetic intensity = 36 / 28 = 1.286 FLOP/byte

3. For each scenario, we'll calculate the compute and memory time to determine the bottleneck:

Case A (200 GFLOPS, 100 GB/s):
* Time for compute = 36 FLOP ÷ (200 × 10⁹ FLOP/s) = 0.18 ns
* Time for memory = 28 bytes ÷ (100 × 10⁹ B/s) = 0.28 ns
* Since memory time > compute time, this kernel is memory-bound

Case B (300 GFLOPS, 250 GB/s):
* Time for compute = 36 FLOP ÷ (300 × 10⁹ FLOP/s) = 0.12 ns
* Time for memory = 28 bytes ÷ (250 × 10⁹ B/s) = 0.112 ns
* Since compute time > memory time, this kernel is compute-bound

Answer:
a. Memory-bound
b. Compute-bound


5.11. Indicate which of the following assignments per streaming multiprocessor is possible. In the case where it is not possible, indicate the limiting factor(s).
a. 4 blocks with 128 threads each and 32 B shared memory per thread on a device with compute capability 1.0.
b. 8 blocks with 128 threads each and 16 B shared memory per thread on a device with compute capability 1.0.
c. 16 blocks with 32 threads each and 64 B shared memory per thread on a device with compute capability 1.0.
d. 2 blocks with 512 threads each and 32 B shared memory per thread on a device with compute capability 1.2.
e. 4 blocks with 256 threads each and 16 B shared memory per thread on a device with compute capability 1.2.
f. 8 blocks with 256 threads each and 8 B shared memory per thread on a device with compute capability 1.2.

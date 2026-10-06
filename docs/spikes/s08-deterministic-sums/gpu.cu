// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

// Spike S8, GPU half: what does a deterministic float sum cost on a GPU?
//
// Same data as the CPU half (50 M doubles over 16 orders of magnitude, mixed
// signs). Each accumulator runs 20 times; we count distinct results.
//
// - atomic:     per-block tree sums, then atomicAdd of each block's result into
//               one double. Block order is up to the scheduler: nondeterministic.
// - fixed tree: per-block tree sums written to an array, then one block sums
//               the array in a fixed pattern. Deterministic.
// - binned:     each value split onto 3 fixed grids from max|x| and n (Demmel
//               and Nguyen); per-level sums are exact, so even atomicAdd on
//               them gives the same bits in any order. Deterministic.
// - superacc:   exact fixed-point limbs accumulated with 64-bit integer
//               atomics; integer addition is associative. Deterministic.
//
// Build: nvcc -O3 -arch=sm_61 gpu.cu -o gpu   (GTX 1080 Ti; use sm_75 for a T4)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; } } while (0)

constexpr int THREADS = 256;
constexpr int LIMBS = 72;

struct Lcg {
  uint64_t s;
  uint64_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s; }
  double unif() { return (double)(next() >> 11) / (double)(1ULL << 53); }
};

__device__ double block_sum(double v) {
  __shared__ double sh[THREADS];
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  double r = sh[0];
  __syncthreads();
  return r;
}

// Each thread sums a fixed, strided set of elements, so per-thread order is fixed.
__global__ void k_partial(const double* x, size_t n, double* block_out) {
  double s = 0;
  for (size_t i = blockIdx.x * (size_t)THREADS + threadIdx.x; i < n; i += (size_t)gridDim.x * THREADS) s += x[i];
  double b = block_sum(s);
  if (threadIdx.x == 0) block_out[blockIdx.x] = b;
}

__global__ void k_atomic(const double* x, size_t n, double* out) {
  double s = 0;
  for (size_t i = blockIdx.x * (size_t)THREADS + threadIdx.x; i < n; i += (size_t)gridDim.x * THREADS) s += x[i];
  double b = block_sum(s);
  if (threadIdx.x == 0) atomicAdd(out, b);
}

__global__ void k_final(const double* parts, int m, double* out) {
  double s = 0;
  for (int i = threadIdx.x; i < m; i += THREADS) s += parts[i];
  double b = block_sum(s);
  if (threadIdx.x == 0) *out = b;
}

__global__ void k_binned(const double* x, size_t n, double s0, double s1, double s2, double* lv) {
  double a0 = 0, a1 = 0, a2 = 0;
  for (size_t i = blockIdx.x * (size_t)THREADS + threadIdx.x; i < n; i += (size_t)gridDim.x * THREADS) {
    double r = x[i], q;
    q = (s0 + r) - s0; a0 += q; r -= q;
    q = (s1 + r) - s1; a1 += q; r -= q;
    q = (s2 + r) - s2; a2 += q;
  }
  // Every partial is exact on its grid, so order does not matter, even atomics.
  atomicAdd(&lv[0], a0);
  atomicAdd(&lv[1], a1);
  atomicAdd(&lv[2], a2);
}

__global__ void k_super(const double* x, size_t n, unsigned long long* limbs) {
  __shared__ long long acc[LIMBS];
  for (int k = threadIdx.x; k < LIMBS; k += THREADS) acc[k] = 0;
  __syncthreads();
  for (size_t i = blockIdx.x * (size_t)THREADS + threadIdx.x; i < n; i += (size_t)gridDim.x * THREADS) {
    unsigned long long bits = __double_as_longlong(x[i]);
    long long e = (bits >> 52) & 0x7ff;
    long long mant = bits & ((1ULL << 52) - 1);
    long long shift = e == 0 ? 0 : (mant |= (1LL << 52), e - 1);
    int limb = shift / 32, off = shift % 32;
    unsigned long long lo = (unsigned long long)mant << off;          // low 64 bits
    unsigned long long hi = off ? (unsigned long long)mant >> (64 - off) : 0;
    long long sign = (bits >> 63) ? -1 : 1;
    atomicAdd((unsigned long long*)&acc[limb], (unsigned long long)(sign * (long long)(lo & 0xffffffffULL)));
    atomicAdd((unsigned long long*)&acc[limb + 1], (unsigned long long)(sign * (long long)(lo >> 32)));
    atomicAdd((unsigned long long*)&acc[limb + 2], (unsigned long long)(sign * (long long)hi));
  }
  __syncthreads();
  for (int k = threadIdx.x; k < LIMBS; k += THREADS) atomicAdd(&limbs[k], (unsigned long long)acc[k]);
}

// Host: normalize limbs and round to double (same as the CPU half's finish()).
double super_finish(const std::vector<long long>& in) {
  std::vector<long long> l(in);
  for (int k = 0; k < LIMBS - 1; k++) { long long c = l[k] >> 32; l[k] -= c << 32; l[k + 1] += c; }
  bool neg = l[LIMBS - 1] < 0;
  if (neg) {
    long long borrow = 0;
    for (int k = 0; k < LIMBS; k++) {
      long long v = -l[k] - borrow;
      if (k < LIMBS - 1) { long long c = v < 0; l[k] = v + (c << 32); borrow = c; } else l[k] = v;
    }
  }
  int top = -1;
  for (int k = LIMBS - 1; k >= 0; k--) if (l[k]) { top = k; break; }
  if (top < 0) return 0.0;
  int lo = top >= 2 ? top - 2 : 0;
  __int128 m = 0;
  for (int k = top; k >= lo; k--) m = (m << 32) | (unsigned long long)l[k];
  bool sticky = false;
  for (int k = 0; k < lo; k++) sticky |= l[k] != 0;
  m = (m << 1) | sticky;
  double r = (double)m * std::ldexp(1.0, 32 * lo - 1074 - 1);
  return neg ? -r : r;
}

int main() {
  const size_t n = 50000000;
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  const int BLOCKS = prop.multiProcessorCount * 8;
  printf("device: %s (sm_%d%d, %d SMs)\n\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
  std::vector<double> h(n);
  Lcg r{7};
  for (auto& v : h) { double mag = std::pow(10.0, r.unif() * 16.0 - 8.0); v = (r.next() & 1) == 0 ? mag : -mag; }
  double mx = 0;
  for (double v : h) mx = std::fmax(mx, std::fabs(v));
  double sig[3], m = mx;
  for (int k = 0; k < 3; k++) { sig[k] = std::ldexp(1.0, (int)std::ceil(std::log2(m * n)) + 1); m = sig[k] * std::ldexp(1.0, -53); }

  double *dx, *dparts, *dout, *dlv;
  unsigned long long* dlimbs;
  CHECK(cudaMalloc(&dx, n * sizeof(double)));
  CHECK(cudaMalloc(&dparts, BLOCKS * sizeof(double)));
  CHECK(cudaMalloc(&dout, sizeof(double)));
  CHECK(cudaMalloc(&dlv, 3 * sizeof(double)));
  CHECK(cudaMalloc(&dlimbs, LIMBS * sizeof(unsigned long long)));
  CHECK(cudaMemcpy(dx, h.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);

  printf("| Accumulator | ms (median of 20) | GB/s | vs atomic | Distinct results in 20 runs | Result |\n|---|---|---|---|---|---|\n");
  double t_atomic = 0;
  for (int which = 0; which < 4; which++) {
    std::vector<float> ts;
    std::set<uint64_t> results;
    double res = 0;
    for (int rep = 0; rep < 21; rep++) {
      cudaEventRecord(a);
      if (which == 0) {
        cudaMemset(dout, 0, sizeof(double));
        k_atomic<<<BLOCKS, THREADS>>>(dx, n, dout);
      } else if (which == 1) {
        k_partial<<<BLOCKS, THREADS>>>(dx, n, dparts);
        k_final<<<1, THREADS>>>(dparts, BLOCKS, dout);
      } else if (which == 2) {
        cudaMemset(dlv, 0, 3 * sizeof(double));
        k_binned<<<BLOCKS, THREADS>>>(dx, n, sig[0], sig[1], sig[2], dlv);
      } else {
        cudaMemset(dlimbs, 0, LIMBS * sizeof(unsigned long long));
        k_super<<<BLOCKS, THREADS>>>(dx, n, dlimbs);
      }
      cudaEventRecord(b);
      CHECK(cudaEventSynchronize(b));
      float ms;
      cudaEventElapsedTime(&ms, a, b);
      if (rep > 0) ts.push_back(ms);  // first run warms up
      if (which <= 1) {
        CHECK(cudaMemcpy(&res, dout, sizeof(double), cudaMemcpyDeviceToHost));
      } else if (which == 2) {
        double lv[3];
        CHECK(cudaMemcpy(lv, dlv, sizeof lv, cudaMemcpyDeviceToHost));
        res = lv[0] + lv[1] + lv[2];
      } else {
        std::vector<long long> l(LIMBS);
        CHECK(cudaMemcpy(l.data(), dlimbs, LIMBS * sizeof(long long), cudaMemcpyDeviceToHost));
        res = super_finish(l);
      }
      uint64_t bits;
      memcpy(&bits, &res, 8);
      results.insert(bits);
    }
    std::sort(ts.begin(), ts.end());
    double med = ts[ts.size() / 2];
    if (which == 0) t_atomic = med;
    const char* names[] = {"atomic (scheduler order)", "fixed tree", "binned, max from facts", "superaccumulator"};
    printf("| %s | %.2f | %.0f | %.1fx | %zu | %.17g |\n", names[which], med, n * 8 / (med * 1e6), med / t_atomic, results.size(), res);
  }
  return 0;
}

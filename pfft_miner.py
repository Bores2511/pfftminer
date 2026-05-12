#!/usr/bin/env python3
"""
PFFT Miner Bot — GPU EDITION (CUDA / OpenCL / CPU Fallback)
Ethereum Mainnet | Contract: 0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB

Prioritas backend:
  1. CUDA    (NVIDIA GPU) — tercepat, jutaan hash/detik
  2. OpenCL  (AMD / Intel GPU) — alternatif GPU
  3. CPU     (multicore fallback)

Setup:
  pip install web3 pycryptodome pyopencl
  pip install pycuda   # khusus NVIDIA (butuh CUDA Toolkit terinstall)

  cp .env.example .env   # isi PRIVATE_KEY
  python3 pfft_miner_gpu.py

Env vars:
  PRIVATE_KEY   = (wajib) private key wallet EVM
  ETH_RPC       = RPC endpoint (default: publicnode)
  BACKEND       = cuda | opencl | cpu  (default: auto-detect)
  GPU_BATCH     = jumlah hash per launch GPU (default: 4_000_000)
  NUM_WORKERS   = jumlah CPU core (fallback, default: semua)
"""

import os
import sys
import time
import struct
import signal
import multiprocessing
from pathlib import Path

# ─── Load .env ──────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ─── Config ──────────────────────────────────────────────────────────────────
CONTRACT    = "0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB"
CHAIN_ID    = 1
RPC         = os.environ.get("ETH_RPC", "https://ethereum-rpc.publicnode.com")
PRIVATE_KEY = os.environ.get("PRIVATE_KEY", "")
GAS_LIMIT   = 200_000
PAUSE_BETWEEN_ROUNDS = 5
BACKEND     = os.environ.get("BACKEND", "auto").lower()
GPU_BATCH   = int(os.environ.get("GPU_BATCH", 4_000_000))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", multiprocessing.cpu_count()))

# ─── Keccak256 (CPU) ─────────────────────────────────────────────────────────
try:
    from Crypto.Hash import keccak as _keccak_mod
except ImportError:
    print("❌  Missing: pip install pycryptodome")
    sys.exit(1)

def keccak256(data: bytes) -> bytes:
    return _keccak_mod.new(digest_bits=256, data=data).digest()


# ════════════════════════════════════════════════════════════════════════════
#  CUDA KERNEL  (NVIDIA GPU)
# ════════════════════════════════════════════════════════════════════════════

CUDA_KERNEL = r"""
/*
 * Keccak-256 GPU kernel untuk PFFT PoW mining
 * Setiap thread menghitung satu keccak256(challenge || nonce_uint256)
 * Jika hash <= target, simpan nonce ke result_nonce[0]
 */

#define KECCAK_ROUNDS 24

typedef unsigned long long uint64_t;
typedef unsigned int       uint32_t;

__constant__ static const uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

__constant__ static const int RHO[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

__constant__ static const int PI[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

__device__ inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ void keccak_f(uint64_t st[25]) {
    uint64_t t, bc[5];
    for (int r = 0; r < 24; r++) {
        // Theta
        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i+5] ^ st[i+10] ^ st[i+15] ^ st[i+20];
        for (int i = 0; i < 5; i++) {
            t = bc[(i+4)%5] ^ rotl64(bc[(i+1)%5], 1);
            for (int j = 0; j < 25; j += 5) st[j+i] ^= t;
        }
        // Rho + Pi
        t = st[1];
        for (int i = 0; i < 24; i++) {
            int j  = PI[i];
            uint64_t tmp = st[j];
            st[j] = rotl64(t, RHO[i]);
            t = tmp;
        }
        // Chi
        for (int j = 0; j < 25; j += 5) {
            uint64_t t0=st[j],t1=st[j+1],t2=st[j+2],t3=st[j+3],t4=st[j+4];
            st[j]   ^= (~t1) & t2;
            st[j+1] ^= (~t2) & t3;
            st[j+2] ^= (~t3) & t4;
            st[j+3] ^= (~t4) & t0;
            st[j+4] ^= (~t0) & t1;
        }
        // Iota
        st[0] ^= RC[r];
    }
}

/*
 * Input layout (64 bytes):
 *   [0..31]  = challenge (bytes32)
 *   [32..63] = nonce as big-endian uint256 (only lower 8 bytes used, upper zero)
 *
 * target_hi / target_lo = 128-bit target (upper and lower half)
 * Kita bandingkan hash[0..15] dengan target (big-endian)
 *
 * result_nonce[0] = -1  =>  belum ketemu
 * result_nonce[0] = N   =>  nonce yang valid
 */
__global__ void pfft_mine(
    const unsigned char *challenge,   // 32 bytes
    unsigned long long   base_nonce,  // nonce awal untuk batch ini
    unsigned long long   target_hi,   // 64-bit upper target
    unsigned long long   target_lo,   // 64-bit lower target (kita cukup compare hi)
    unsigned long long  *result_nonce,
    unsigned long long  *hash_count
) {
    unsigned long long nonce = base_nonce + (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Build input: challenge (32 bytes) + nonce as uint256 big-endian (32 bytes)
    uint64_t st[25];
    memset(st, 0, sizeof(st));

    // Load challenge into st[0..3] (32 bytes = 4 x uint64, little-endian lanes for keccak)
    unsigned char buf[136]; // keccak-256 rate = 136 bytes
    memset(buf, 0, sizeof(buf));
    for (int i = 0; i < 32; i++) buf[i] = challenge[i];

    // Nonce as big-endian uint256: only lower 8 bytes
    buf[32+24] = (unsigned char)((nonce >> 56) & 0xff);
    buf[32+25] = (unsigned char)((nonce >> 48) & 0xff);
    buf[32+26] = (unsigned char)((nonce >> 40) & 0xff);
    buf[32+27] = (unsigned char)((nonce >> 32) & 0xff);
    buf[32+28] = (unsigned char)((nonce >> 24) & 0xff);
    buf[32+29] = (unsigned char)((nonce >> 16) & 0xff);
    buf[32+30] = (unsigned char)((nonce >>  8) & 0xff);
    buf[32+31] = (unsigned char)( nonce        & 0xff);

    // Keccak-256 padding (rate=136, domain=0x01)
    buf[64]  ^= 0x01;
    buf[135] ^= 0x80;

    // Absorb into state (little-endian uint64)
    for (int i = 0; i < 17; i++) {
        uint64_t w = 0;
        for (int b = 0; b < 8; b++)
            w |= ((uint64_t)buf[i*8 + b]) << (b * 8);
        st[i] ^= w;
    }

    keccak_f(st);

    // Read first 16 bytes of hash (big-endian) for comparison
    // st[0] is bytes 0-7 in little-endian
    unsigned long long h0 = 0, h1 = 0;
    for (int b = 0; b < 8; b++) {
        h0 = (h0 << 8) | ((st[0] >> (b*8)) & 0xff);
        h1 = (h1 << 8) | ((st[1] >> (b*8)) & 0xff);
    }

    // Compare: hash (big-endian 256-bit) <= target
    // Kita cukup compare 128 bit pertama (h0 dan h1) vs target_hi dan target_lo
    bool valid = false;
    if (h0 < target_hi) {
        valid = true;
    } else if (h0 == target_hi && h1 <= target_lo) {
        valid = true;
    }

    if (valid) {
        // Atomic: simpan nonce pertama yang valid
        unsigned long long old = atomicCAS(result_nonce, (unsigned long long)-1, nonce);
        (void)old;
    }

    // Count hashes (per warp, untuk kurangi atomic contention)
    if (threadIdx.x % 32 == 0)
        atomicAdd(hash_count, 32ULL);
}
"""


def detect_backend():
    """Auto-detect GPU backend terbaik yang tersedia."""
    if BACKEND in ("cuda", "auto"):
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as drv
            n = drv.Device.count()
            if n > 0:
                print(f"🟢  CUDA detected: {n} GPU(s)")
                return "cuda"
        except Exception as e:
            if BACKEND == "cuda":
                print(f"❌  CUDA tidak tersedia: {e}")
                sys.exit(1)

    if BACKEND in ("opencl", "auto"):
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            gpu_devs  = []
            for p in platforms:
                gpu_devs += p.get_devices(cl.device_type.GPU)
            if gpu_devs:
                print(f"🟡  OpenCL detected: {len(gpu_devs)} GPU device(s)")
                return "opencl"
        except Exception:
            pass

    if BACKEND == "opencl":
        print("❌  OpenCL tidak tersedia.")
        sys.exit(1)

    print(f"⚪  Tidak ada GPU — fallback ke {NUM_WORKERS} CPU core(s)")
    return "cpu"


# ════════════════════════════════════════════════════════════════════════════
#  CUDA SOLVER
# ════════════════════════════════════════════════════════════════════════════

def solve_cuda(challenge: bytes, target: int) -> tuple:
    import pycuda.autoinit          # noqa: F401
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    import numpy as np

    mod          = SourceModule(CUDA_KERNEL, no_extern_c=True)
    pfft_mine_fn = mod.get_function("pfft_mine")

    # Hitung optimal block/grid size
    dev            = drv.Device(0)
    max_threads    = dev.get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK)
    threads        = min(512, max_threads)
    sm_count       = dev.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)
    blocks_per_sm  = 8
    blocks         = sm_count * blocks_per_sm
    batch_per_call = blocks * threads  # total thread per launch

    print(f"  🎮 GPU: {dev.name()}")
    print(f"  🔢 Threads/block: {threads} | Blocks: {blocks} | Batch: {batch_per_call:,}/launch")

    # Siapkan target sebagai dua uint64 (upper 128-bit)
    target_bytes = target.to_bytes(32, 'big')
    target_hi    = int.from_bytes(target_bytes[0:8],  'big')
    target_lo    = int.from_bytes(target_bytes[8:16], 'big')

    challenge_gpu  = drv.mem_alloc(32)
    drv.memcpy_htod(challenge_gpu, challenge)

    result_nonce   = np.array([(2**64 - 1)], dtype=np.uint64)
    hash_count_gpu = np.zeros(1, dtype=np.uint64)
    result_gpu     = drv.mem_alloc(8)
    hashcnt_gpu    = drv.mem_alloc(8)
    drv.memcpy_htod(result_gpu,  result_nonce)
    drv.memcpy_htod(hashcnt_gpu, hash_count_gpu)

    start       = time.time()
    last_report = start
    last_hashes = 0
    base_nonce  = np.uint64(0)

    print(f"  ⛏️  Mining dengan CUDA GPU (batch {GPU_BATCH:,}) ...")

    while True:
        pfft_mine_fn(
            challenge_gpu,
            base_nonce,
            np.uint64(target_hi),
            np.uint64(target_lo),
            result_gpu,
            hashcnt_gpu,
            block=(threads, 1, 1),
            grid=(blocks, 1)
        )
        drv.Context.synchronize()

        drv.memcpy_dtoh(result_nonce, result_gpu)
        if result_nonce[0] != (2**64 - 1):
            break

        base_nonce = np.uint64(int(base_nonce) + batch_per_call)

        # Progress report
        now = time.time()
        if now - last_report >= 3.0:
            drv.memcpy_dtoh(hash_count_gpu, hashcnt_gpu)
            total   = int(hash_count_gpu[0])
            window  = total - last_hashes
            elapsed = now - start
            rate    = window / (now - last_report)
            avg     = total / elapsed if elapsed > 0 else 0
            print(
                f"  🎮  CUDA | {total:,} hashes | "
                f"{rate/1e6:.2f} MH/s (now) | "
                f"{avg/1e6:.2f} MH/s (avg) | "
                f"{elapsed:.0f}s",
                end='\r'
            )
            last_report = now
            last_hashes = total

    nonce   = int(result_nonce[0])
    elapsed = time.time() - start
    drv.memcpy_dtoh(hash_count_gpu, hashcnt_gpu)
    total   = int(hash_count_gpu[0])
    rate    = total / elapsed if elapsed > 0 else 0

    print(f"\n  ✅ SOLVED  nonce={nonce} | {total:,} hashes | {rate/1e6:.2f} MH/s | {elapsed:.1f}s | CUDA")

    # Verifikasi lokal
    buf = bytearray(challenge) + bytearray(32)
    struct.pack_into('>QQQQ', buf, 32, 0, 0, 0, nonce)
    h = keccak256(bytes(buf))
    return nonce, h


# ════════════════════════════════════════════════════════════════════════════
#  OPENCL SOLVER
# ════════════════════════════════════════════════════════════════════════════

OPENCL_KERNEL = r"""
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

typedef ulong  uint64_t;
typedef uint   uint32_t;
typedef uchar  uint8_t;

__constant uint64_t RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808aUL, 0x8000000080008000UL,
    0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008aUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL
};

__constant int RHO[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

__constant int PI[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccak_f(uint64_t st[25]) {
    uint64_t t, bc[5];
    for (int r = 0; r < 24; r++) {
        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i+5] ^ st[i+10] ^ st[i+15] ^ st[i+20];
        for (int i = 0; i < 5; i++) {
            t = bc[(i+4)%5] ^ rotl64(bc[(i+1)%5], 1);
            for (int j = 0; j < 25; j += 5) st[j+i] ^= t;
        }
        t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = PI[i];
            uint64_t tmp = st[j];
            st[j] = rotl64(t, RHO[i]);
            t = tmp;
        }
        for (int j = 0; j < 25; j += 5) {
            uint64_t t0=st[j],t1=st[j+1],t2=st[j+2],t3=st[j+3],t4=st[j+4];
            st[j]   ^= (~t1) & t2;
            st[j+1] ^= (~t2) & t3;
            st[j+2] ^= (~t3) & t4;
            st[j+3] ^= (~t4) & t0;
            st[j+4] ^= (~t0) & t1;
        }
        st[0] ^= RC[r];
    }
}

__kernel void pfft_mine(
    __global const uint8_t *challenge,
    ulong  base_nonce,
    ulong  target_hi,
    ulong  target_lo,
    __global ulong *result_nonce,
    __global ulong *hash_count
) {
    ulong nonce = base_nonce + get_global_id(0);

    uint8_t buf[136];
    for (int i = 0; i < 136; i++) buf[i] = 0;
    for (int i = 0; i < 32; i++) buf[i] = challenge[i];

    buf[32+24] = (uint8_t)((nonce >> 56) & 0xff);
    buf[32+25] = (uint8_t)((nonce >> 48) & 0xff);
    buf[32+26] = (uint8_t)((nonce >> 40) & 0xff);
    buf[32+27] = (uint8_t)((nonce >> 32) & 0xff);
    buf[32+28] = (uint8_t)((nonce >> 24) & 0xff);
    buf[32+29] = (uint8_t)((nonce >> 16) & 0xff);
    buf[32+30] = (uint8_t)((nonce >>  8) & 0xff);
    buf[32+31] = (uint8_t)( nonce        & 0xff);

    buf[64]  ^= 0x01;
    buf[135] ^= 0x80;

    uint64_t st[25];
    for (int i = 0; i < 25; i++) st[i] = 0;

    for (int i = 0; i < 17; i++) {
        uint64_t w = 0;
        for (int b = 0; b < 8; b++)
            w |= ((uint64_t)buf[i*8 + b]) << (b * 8);
        st[i] ^= w;
    }

    keccak_f(st);

    uint64_t h0 = 0, h1 = 0;
    for (int b = 0; b < 8; b++) {
        h0 = (h0 << 8) | ((st[0] >> (b*8)) & 0xff);
        h1 = (h1 << 8) | ((st[1] >> (b*8)) & 0xff);
    }

    bool valid = false;
    if (h0 < target_hi) valid = true;
    else if (h0 == target_hi && h1 <= target_lo) valid = true;

    if (valid) {
        atom_cmpxchg(result_nonce, (ulong)(-1), nonce);
    }

    if (get_local_id(0) == 0)
        atom_add(hash_count, (ulong)get_local_size(0));
}
"""


def solve_opencl(challenge: bytes, target: int) -> tuple:
    import pyopencl as cl
    import numpy as np

    platform = cl.get_platforms()[0]
    gpu_devs = platform.get_devices(cl.device_type.GPU)
    device   = gpu_devs[0]
    ctx      = cl.Context([device])
    queue    = cl.CommandQueue(ctx)

    print(f"  🎮 OpenCL GPU: {device.name.strip()}")

    prog  = cl.Program(ctx, OPENCL_KERNEL).build()
    mf    = cl.mem_flags

    target_bytes = target.to_bytes(32, 'big')
    target_hi    = int.from_bytes(target_bytes[0:8],  'big')
    target_lo    = int.from_bytes(target_bytes[8:16], 'big')

    challenge_buf  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bytes(challenge))
    result_np      = np.array([(2**64 - 1)], dtype=np.uint64)
    hashcnt_np     = np.zeros(1, dtype=np.uint64)
    result_buf     = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=result_np)
    hashcnt_buf    = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hashcnt_np)

    max_wu      = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    local_size  = min(256, max_wu)
    global_size = GPU_BATCH

    start       = time.time()
    last_report = start
    last_hashes = 0
    base_nonce  = np.uint64(0)

    print(f"  ⛏️  Mining dengan OpenCL GPU (batch {GPU_BATCH:,}) ...")

    while True:
        prog.pfft_mine(
            queue, (global_size,), (local_size,),
            challenge_buf,
            np.uint64(base_nonce),
            np.uint64(target_hi),
            np.uint64(target_lo),
            result_buf,
            hashcnt_buf
        )
        queue.finish()

        cl.enqueue_copy(queue, result_np, result_buf)
        if result_np[0] != (2**64 - 1):
            break

        base_nonce = np.uint64(int(base_nonce) + global_size)

        now = time.time()
        if now - last_report >= 3.0:
            cl.enqueue_copy(queue, hashcnt_np, hashcnt_buf)
            total   = int(hashcnt_np[0])
            window  = total - last_hashes
            elapsed = now - start
            rate    = window / (now - last_report)
            avg     = total / elapsed if elapsed > 0 else 0
            print(
                f"  🎮  OpenCL | {total:,} hashes | "
                f"{rate/1e6:.2f} MH/s (now) | "
                f"{avg/1e6:.2f} MH/s (avg) | "
                f"{elapsed:.0f}s",
                end='\r'
            )
            last_report = now
            last_hashes = total

    nonce   = int(result_np[0])
    elapsed = time.time() - start
    cl.enqueue_copy(queue, hashcnt_np, hashcnt_buf)
    total   = int(hashcnt_np[0])
    rate    = total / elapsed if elapsed > 0 else 0

    print(f"\n  ✅ SOLVED  nonce={nonce} | {total:,} hashes | {rate/1e6:.2f} MH/s | {elapsed:.1f}s | OpenCL")

    buf = bytearray(challenge) + bytearray(32)
    struct.pack_into('>QQQQ', buf, 32, 0, 0, 0, nonce)
    h = keccak256(bytes(buf))
    return nonce, h


# ════════════════════════════════════════════════════════════════════════════
#  CPU FALLBACK SOLVER (multicore)
# ════════════════════════════════════════════════════════════════════════════

def _cpu_worker(worker_id, num_workers, challenge, target,
                result_queue, stop_event, counter):
    nonce = worker_id
    buf   = bytearray(challenge) + bytearray(32)
    local_count = 0
    while not stop_event.is_set():
        struct.pack_into('>QQQQ', buf, 32, 0, 0, 0, nonce)
        h     = keccak256(bytes(buf))
        h_int = int.from_bytes(h, 'big')
        if h_int <= target:
            result_queue.put((nonce, bytes(h)))
            stop_event.set()
            return
        nonce += num_workers
        local_count += 1
        if local_count % 5_000 == 0:
            with counter.get_lock():
                counter.value += 5_000


def solve_cpu(challenge: bytes, target: int) -> tuple:
    result_queue = multiprocessing.Queue()
    stop_event   = multiprocessing.Event()
    counter      = multiprocessing.Value('Q', 0)
    workers = []
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(
            target=_cpu_worker,
            args=(i, NUM_WORKERS, challenge, target,
                  result_queue, stop_event, counter),
            daemon=True
        )
        p.start()
        workers.append(p)

    start       = time.time()
    last_report = start
    last_count  = 0
    result      = None

    print(f"  ⛏️  Mining dengan {NUM_WORKERS} CPU core(s) ...")

    while result is None:
        time.sleep(0.5)
        if not result_queue.empty():
            result = result_queue.get()
            break
        now = time.time()
        if now - last_report >= 5.0:
            total   = counter.value
            window  = total - last_count
            elapsed = now - start
            rate    = window / (now - last_report)
            avg     = total / elapsed if elapsed > 0 else 0
            print(
                f"  ⚙️  CPU | {total:,} hashes | "
                f"{rate/1000:.1f} kH/s (now) | "
                f"{avg/1000:.1f} kH/s (avg) | "
                f"{elapsed:.0f}s",
                end='\r'
            )
            last_report = now
            last_count  = total

    stop_event.set()
    for p in workers:
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()

    nonce, h = result
    elapsed   = time.time() - start
    total     = counter.value
    rate      = total / elapsed if elapsed > 0 else 0
    print(f"\n  ✅ SOLVED  nonce={nonce} | {total:,} hashes | {rate/1000:.1f} kH/s | {elapsed:.1f}s | CPU")
    return nonce, h


# ════════════════════════════════════════════════════════════════════════════
#  CONTRACT HELPERS
# ════════════════════════════════════════════════════════════════════════════

ABI = [
    {"inputs":[],"name":"currentPowHexZeros","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"totalMinted","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"MAX_SUPPLY","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"requested","type":"uint256"}],"name":"calculateActualMint","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"user","type":"address"}],"name":"currentPowChallenge","outputs":[{"type":"bytes32"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"user","type":"address"},{"name":"powNonce","type":"uint256"}],"name":"isValidPow","outputs":[{"type":"bool"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"powNonce","type":"uint256"}],"name":"freeMint","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"account","type":"address"}],"name":"mintedByAddress","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"type":"uint256"}],"stateMutability":"view","type":"function"},
]


def load_contract(w3):
    from web3 import Web3
    return w3.eth.contract(address=Web3.to_checksum_address(CONTRACT), abi=ABI)


def get_status(contract, w3, wallet_addr):
    hex_zeros     = contract.functions.currentPowHexZeros().call()
    total_minted  = contract.functions.totalMinted().call()
    max_supply    = contract.functions.MAX_SUPPLY().call()
    next_mint     = contract.functions.calculateActualMint(w3.to_wei(1000, 'ether')).call()
    wallet_minted = contract.functions.mintedByAddress(wallet_addr).call()
    wallet_bal    = contract.functions.balanceOf(wallet_addr).call()
    target        = (2**256 - 1) >> (hex_zeros * 4)
    progress      = total_minted * 10_000 / max_supply / 100
    return {
        "hex_zeros": hex_zeros,
        "difficulty_bits": hex_zeros * 4,
        "total_minted": total_minted,
        "max_supply": max_supply,
        "next_mint": next_mint,
        "wallet_minted": wallet_minted,
        "wallet_bal": wallet_bal,
        "target": target,
        "progress": progress,
    }


def get_challenge(contract, wallet_addr):
    c = contract.functions.currentPowChallenge(wallet_addr).call()
    return c if isinstance(c, bytes) else c.to_bytes(32, 'big')


def submit_mint(w3, wallet, contract, nonce):
    try:
        fn = contract.functions.freeMint(nonce)
        tx = fn.build_transaction({
            'from':    wallet.address,
            'nonce':   w3.eth.get_transaction_count(wallet.address),
            'chainId': CHAIN_ID,
            'gas':     GAS_LIMIT,
        })
        if 'maxFeePerGas' not in tx and 'maxPriorityFeePerGas' not in tx:
            tx['gasPrice'] = w3.eth.gas_price
        signed  = wallet.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"  📤 TX broadcast → https://etherscan.io/tx/0x{tx_hash.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        if receipt.status == 1:
            print(f"  ✅ MINT SUCCESS | Block #{receipt.blockNumber} | Gas used: {receipt.gasUsed:,}")
            return True
        else:
            print(f"  ❌ TX REVERTED | Gas used: {receipt.gasUsed:,}")
            return False
    except Exception as e:
        print(f"  ❌ TX error: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════════
#  SIGNAL HANDLING
# ════════════════════════════════════════════════════════════════════════════

running = True

def handle_signal(sig, frame):
    global running
    print("\n\n  ⚠️  Interrupt received — finishing current round then stopping...")
    running = False

signal.signal(signal.SIGINT,  handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    from web3 import Web3
    from eth_account import Account

    backend = detect_backend()

    # Pilih solver
    if backend == "cuda":
        solve = solve_cuda
    elif backend == "opencl":
        solve = solve_opencl
    else:
        solve = solve_cpu

    bar = "═" * 64
    print(bar)
    print("  ⛏️   PFFT Miner Bot — GPU EDITION")
    print(f"  Backend   : {backend.upper()}")
    print(f"  Contract  : {CONTRACT}")
    print(f"  RPC       : {RPC}")
    print(bar)

    # Connect
    w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        print("❌  Tidak bisa connect ke RPC endpoint")
        sys.exit(1)
    print(f"✅  RPC connected    | Block #{w3.eth.block_number:,}")

    # Wallet
    pk = PRIVATE_KEY.strip()
    if not pk or pk == "your_private_key_here":
        print("❌  PRIVATE_KEY belum diset!")
        print("    Salin .env.example → .env dan isi private key kamu.")
        sys.exit(1)
    if not pk.startswith('0x'):
        pk = '0x' + pk

    wallet  = Account.from_key(pk)
    eth_bal = w3.eth.get_balance(wallet.address) / 1e18
    print(f"✅  Wallet loaded    | {wallet.address}")
    print(f"💰  ETH balance      | {eth_bal:.6f} ETH")
    if eth_bal < 0.00005:
        print("⚠️  ETH balance rendah! Top up ~0.001 ETH untuk gas.")

    # Contract status
    contract = load_contract(w3)
    s        = get_status(contract, w3, wallet.address)

    print(f"\n📊  Contract Status")
    print(f"    Supply       : {s['total_minted']/1e18:>12,.0f} / {s['max_supply']/1e18:,.0f} PFFT  ({s['progress']:.2f}%)")
    print(f"    Next mint    : ~{s['next_mint']/1e18:,.2f} PFFT")
    print(f"    Difficulty   : {s['hex_zeros']} hex zeros  ({s['difficulty_bits']}-bit PoW)")
    print(f"    Wallet minted: {s['wallet_minted']/1e18:>10,.2f} / 10,000.00 PFFT")
    print(f"    Wallet bal   : {s['wallet_bal']/1e18:>10,.2f} PFFT")

    # Mining loop
    round_num          = 0
    total_minted_count = 0
    total_pfft_earned  = 0.0
    global_start       = time.time()

    while running:
        round_num += 1
        ts = time.strftime("%H:%M:%S")
        print(f"\n{'─'*64}")
        print(f"  Round #{round_num}  [{ts}]  |  Backend: {backend.upper()}")
        print(f"{'─'*64}")

        try:
            s = get_status(contract, w3, wallet.address)
            print(
                f"  Supply : {s['total_minted']/1e18:,.0f} PFFT  ({s['progress']:.2f}%)  |  "
                f"Diff: {s['difficulty_bits']}-bit  |  "
                f"Next: ~{s['next_mint']/1e18:,.2f} PFFT"
            )

            if s['total_minted'] >= s['max_supply']:
                print("  🏁 Max supply reached! Mining selesai.")
                break
            if s['wallet_minted'] >= 10_000 * 1e18:
                print("  🏁 Wallet cap (10,000 PFFT) reached!")
                break

        except Exception as e:
            print(f"  ⚠️  Status fetch failed: {e}")
            time.sleep(15)
            continue

        challenge = get_challenge(contract, wallet.address)
        print(f"  Challenge: 0x{challenge.hex()[:16]}...")

        t0          = time.time()
        nonce, h    = solve(challenge, s['target'])
        mining_time = time.time() - t0

        try:
            is_valid = contract.functions.isValidPow(wallet.address, nonce).call()
            if not is_valid:
                print("  ⚠️  On-chain validation failed. Re-mining...")
                continue
            print(f"  ✔️  On-chain validation passed")
        except Exception as e:
            print(f"  ⚠️  Validation check error: {e} — submitting anyway...")

        success = submit_mint(w3, wallet, contract, nonce)
        if success:
            total_minted_count += 1
            earned              = s['next_mint'] / 1e18
            total_pfft_earned  += earned
            try:
                new_bal = contract.functions.balanceOf(wallet.address).call()
                print(f"  💰 Earned   : +{earned:,.2f} PFFT")
                print(f"  💰 Balance  : {new_bal/1e18:,.2f} PFFT")
            except Exception:
                print(f"  💰 Earned   : +{earned:,.2f} PFFT")

        elapsed = time.time() - global_start
        avg_min = (elapsed / 60 / total_minted_count) if total_minted_count > 0 else 0
        print(
            f"\n  📈 Session  : {total_minted_count} mints | "
            f"{total_pfft_earned:,.2f} PFFT | "
            f"{elapsed/60:.1f} min | ~{avg_min:.1f} min/mint"
        )

        if running:
            print(f"  ⏳ {PAUSE_BETWEEN_ROUNDS}s cooldown...")
            time.sleep(PAUSE_BETWEEN_ROUNDS)

    # Summary
    elapsed = time.time() - global_start
    print(f"\n{bar}")
    print(f"  📋  Session Summary")
    print(f"  {'Mints':<18}: {total_minted_count}")
    print(f"  {'PFFT earned':<18}: {total_pfft_earned:,.2f}")
    print(f"  {'Runtime':<18}: {elapsed/60:.1f} min")
    print(f"  {'Backend':<18}: {backend.upper()}")
    print(bar)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

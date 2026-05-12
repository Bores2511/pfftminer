#!/usr/bin/env python3
"""
PFFT Miner Bot — Pow Free Fair Token (MULTI-CORE EDITION)
Ethereum Mainnet | Contract: 0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB

Automatically uses ALL available CPU cores on your VPS for maximum hashrate.

Usage:
  cp .env.example .env        # set PRIVATE_KEY
  python3 pfft_miner.py       # auto-detect all cores

  # Optional: override core count
  NUM_WORKERS=4 python3 pfft_miner.py
"""

import os
import sys
import time
import struct
import signal
import multiprocessing
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env file (no external dependency)
# ---------------------------------------------------------------------------
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONTRACT    = "0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB"
CHAIN_ID    = 1
RPC         = os.environ.get("ETH_RPC", "https://ethereum-rpc.publicnode.com")
PRIVATE_KEY = os.environ.get("PRIVATE_KEY", "")
GAS_LIMIT   = 200_000
PAUSE_BETWEEN_ROUNDS = 5

# Auto-detect ALL CPU cores — override with NUM_WORKERS env var
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", multiprocessing.cpu_count()))


# ---------------------------------------------------------------------------
# Keccak256 — uses pycryptodome C extension (fast)
# ---------------------------------------------------------------------------
try:
    from Crypto.Hash import keccak as _keccak_mod
except ImportError:
    print("❌  Missing dependency!")
    print("    pip install web3 pycryptodome")
    sys.exit(1)


def keccak256(data: bytes) -> bytes:
    return _keccak_mod.new(digest_bits=256, data=data).digest()


# ---------------------------------------------------------------------------
# Worker — runs in a separate process, owns a slice of nonce space
# ---------------------------------------------------------------------------
def worker_solve(worker_id: int,
                 num_workers: int,
                 challenge: bytes,
                 target: int,
                 result_queue: multiprocessing.Queue,
                 stop_event: multiprocessing.Event,
                 counter: multiprocessing.Value):
    """
    Each worker starts at worker_id and steps by num_workers, so the
    entire nonce space is partitioned evenly with no overlap.
    """
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

        nonce       += num_workers
        local_count += 1

        # Batch-update shared counter every 5000 hashes to reduce lock contention
        if local_count % 5_000 == 0:
            with counter.get_lock():
                counter.value += 5_000


# ---------------------------------------------------------------------------
# Multi-core PoW solver — orchestrates all workers
# ---------------------------------------------------------------------------
def solve_pow_multicore(challenge: bytes,
                        target: int,
                        num_workers: int = NUM_WORKERS) -> tuple:
    """
    Spawns num_workers processes. Returns (nonce, hash_bytes) when solved.
    Prints a live hashrate ticker every 5 seconds.
    """
    result_queue = multiprocessing.Queue()
    stop_event   = multiprocessing.Event()
    counter      = multiprocessing.Value('Q', 0)   # unsigned long long

    workers = []
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=worker_solve,
            args=(i, num_workers, challenge, target,
                  result_queue, stop_event, counter),
            daemon=True,
        )
        p.start()
        workers.append(p)

    start       = time.time()
    last_report = start
    last_count  = 0
    result      = None

    while result is None:
        time.sleep(0.5)

        if not result_queue.empty():
            result = result_queue.get()
            break

        now = time.time()
        if now - last_report >= 5.0:
            elapsed     = now - start
            total       = counter.value
            window_hash = total - last_count
            window_sec  = now - last_report
            rate        = window_hash / window_sec if window_sec > 0 else 0
            rate_total  = total / elapsed if elapsed > 0 else 0

            print(
                f"  ⛏️  [{num_workers} cores] "
                f"{total:,} hashes | "
                f"{rate/1000:.1f} kH/s (now) | "
                f"{rate_total/1000:.1f} kH/s (avg) | "
                f"{elapsed:.0f}s",
                end='\r'
            )
            last_report = now
            last_count  = total

    # Clean up workers
    stop_event.set()
    for p in workers:
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()

    nonce, h = result
    elapsed   = time.time() - start
    total     = counter.value
    rate      = total / elapsed if elapsed > 0 else 0

    print(
        f"\n  ✅ SOLVED  nonce={nonce} | "
        f"{total:,} hashes | "
        f"{rate/1000:.1f} kH/s | "
        f"{elapsed:.1f}s | "
        f"{num_workers} cores"
    )
    return nonce, h


# ---------------------------------------------------------------------------
# Contract helpers
# ---------------------------------------------------------------------------
ABI = [
    {
        "inputs": [],
        "name": "currentPowHexZeros",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalMinted",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "MAX_SUPPLY",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "requested", "type": "uint256"}],
        "name": "calculateActualMint",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "user", "type": "address"}],
        "name": "currentPowChallenge",
        "outputs": [{"type": "bytes32"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "user",     "type": "address"},
            {"name": "powNonce", "type": "uint256"}
        ],
        "name": "isValidPow",
        "outputs": [{"type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "powNonce", "type": "uint256"}],
        "name": "freeMint",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "user", "type": "address"}],
        "name": "mintedByAddress",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
]


def load_contract(w3):
    return w3.eth.contract(
        address=w3.to_checksum_address(CONTRACT),
        abi=ABI
    )


def get_status(contract, w3, wallet_addr: str) -> dict:
    hex_zeros     = contract.functions.currentPowHexZeros().call()
    total_minted  = contract.functions.totalMinted().call()
    max_supply    = contract.functions.MAX_SUPPLY().call()
    next_mint     = contract.functions.calculateActualMint(
                        w3.to_wei(1000, 'ether')).call()
    wallet_minted = contract.functions.mintedByAddress(wallet_addr).call()
    wallet_bal    = contract.functions.balanceOf(wallet_addr).call()
    target        = (2**256 - 1) >> (hex_zeros * 4)
    progress      = total_minted * 10_000 / max_supply / 100

    return {
        "hex_zeros":       hex_zeros,
        "difficulty_bits": hex_zeros * 4,
        "total_minted":    total_minted,
        "max_supply":      max_supply,
        "next_mint":       next_mint,
        "wallet_minted":   wallet_minted,
        "wallet_bal":      wallet_bal,
        "target":          target,
        "progress":        progress,
    }


def get_challenge(contract, wallet_addr: str) -> bytes:
    c = contract.functions.currentPowChallenge(wallet_addr).call()
    return c if isinstance(c, bytes) else c.to_bytes(32, 'big')


def submit_mint(w3, wallet, contract, nonce: int) -> bool:
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
            print(
                f"  ✅ MINT SUCCESS | "
                f"Block #{receipt.blockNumber} | "
                f"Gas used: {receipt.gasUsed:,}"
            )
            return True
        else:
            print(f"  ❌ TX REVERTED | Gas used: {receipt.gasUsed:,}")
            return False

    except Exception as e:
        print(f"  ❌ TX error: {e}")
        return False


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
running = True

def handle_signal(sig, frame):
    global running
    print("\n\n  ⚠️  Interrupt received — finishing current round then stopping...")
    running = False


signal.signal(signal.SIGINT,  handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from web3 import Web3
    from eth_account import Account

    bar = "═" * 62

    print(bar)
    print("  ⛏️   PFFT Miner Bot — MULTI-CORE EDITION")
    print(f"  Contract  : {CONTRACT}")
    print(f"  RPC       : {RPC}")
    print(f"  CPU Cores : {NUM_WORKERS}  (set NUM_WORKERS=N to override)")
    print(bar)

    # ── Connect ──────────────────────────────────────────────────────────────
    w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        print("❌  Cannot connect to RPC endpoint")
        sys.exit(1)
    print(f"✅  RPC connected    | Block #{w3.eth.block_number:,}")

    # ── Wallet ───────────────────────────────────────────────────────────────
    pk = PRIVATE_KEY.strip()
    if not pk or pk == "your_private_key_here":
        print("❌  PRIVATE_KEY not set!")
        print("    Copy .env.example → .env and fill in your private key.")
        sys.exit(1)
    if not pk.startswith('0x'):
        pk = '0x' + pk

    wallet  = Account.from_key(pk)
    eth_bal = w3.eth.get_balance(wallet.address) / 1e18
    print(f"✅  Wallet loaded    | {wallet.address}")
    print(f"💰  ETH balance      | {eth_bal:.6f} ETH")
    if eth_bal < 0.00005:
        print("⚠️  Low ETH balance! Top up ~0.001 ETH for gas.")

    # ── Contract status ───────────────────────────────────────────────────────
    contract = load_contract(w3)
    s        = get_status(contract, w3, wallet.address)

    print(f"\n📊  Contract Status")
    print(f"    Supply       : {s['total_minted']/1e18:>12,.0f} / {s['max_supply']/1e18:,.0f} PFFT  ({s['progress']:.2f}%)")
    print(f"    Next mint    : ~{s['next_mint']/1e18:,.2f} PFFT")
    print(f"    Difficulty   : {s['hex_zeros']} hex zeros  ({s['difficulty_bits']}-bit PoW)")
    print(f"    Wallet minted: {s['wallet_minted']/1e18:>10,.2f} / 10,000.00 PFFT")
    print(f"    Wallet bal   : {s['wallet_bal']/1e18:>10,.2f} PFFT")

    # ── Mining loop ───────────────────────────────────────────────────────────
    round_num          = 0
    total_minted_count = 0
    total_pfft_earned  = 0.0
    global_start       = time.time()

    while running:
        round_num += 1
        ts = time.strftime("%H:%M:%S")
        print(f"\n{'─'*62}")
        print(f"  Round #{round_num}  [{ts}]  |  {NUM_WORKERS} cores active")
        print(f"{'─'*62}")

        # Refresh contract state
        try:
            s = get_status(contract, w3, wallet.address)
            print(
                f"  Supply   : {s['total_minted']/1e18:,.0f} PFFT  ({s['progress']:.2f}%)  |  "
                f"Diff: {s['difficulty_bits']}-bit  |  "
                f"Next: ~{s['next_mint']/1e18:,.2f} PFFT"
            )

            if s['total_minted'] >= s['max_supply']:
                print("  🏁 Max supply reached! Mining complete.")
                break
            if s['wallet_minted'] >= 10_000 * 1e18:
                print("  🏁 Wallet cap (10,000 PFFT) reached!")
                break

        except Exception as e:
            print(f"  ⚠️  Status fetch failed: {e}")
            print("  ⏳ Retrying in 15s...")
            time.sleep(15)
            continue

        # Get PoW challenge
        challenge = get_challenge(contract, wallet.address)
        print(f"  Challenge: 0x{challenge.hex()[:16]}...")

        # ── Solve PoW ─────────────────────────────────────────────────────
        print(f"  ⛏️  Solving {s['difficulty_bits']}-bit PoW across {NUM_WORKERS} CPU cores...")
        t0          = time.time()
        nonce, h    = solve_pow_multicore(challenge, s['target'], NUM_WORKERS)
        mining_time = time.time() - t0

        # Verify nonce on-chain before submitting
        try:
            is_valid = contract.functions.isValidPow(wallet.address, nonce).call()
            if not is_valid:
                print("  ⚠️  On-chain validation failed (challenge may have changed). Re-mining...")
                continue
            print(f"  ✔️  On-chain validation passed")
        except Exception as e:
            print(f"  ⚠️  Validation check error: {e} — submitting anyway...")

        # Submit mint tx
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

        # Session stats
        elapsed = time.time() - global_start
        avg_min = (elapsed / 60 / total_minted_count) if total_minted_count > 0 else 0
        print(
            f"\n  📈 Session  : {total_minted_count} mints | "
            f"{total_pfft_earned:,.2f} PFFT | "
            f"{elapsed/60:.1f} min elapsed | "
            f"~{avg_min:.1f} min/mint"
        )

        if running:
            print(f"  ⏳ {PAUSE_BETWEEN_ROUNDS}s cooldown...")
            time.sleep(PAUSE_BETWEEN_ROUNDS)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - global_start
    print(f"\n{bar}")
    print(f"  📋  Session Summary")
    print(f"  {'Mints':<16}: {total_minted_count}")
    print(f"  {'PFFT earned':<16}: {total_pfft_earned:,.2f}")
    print(f"  {'Runtime':<16}: {elapsed/60:.1f} min")
    print(f"  {'CPU cores used':<16}: {NUM_WORKERS}")
    print(bar)


if __name__ == "__main__":
    multiprocessing.freeze_support()   # needed on Windows
    main()

# PFFT Miner Bot ⛏️ — Multi-Core Edition

Automated multi-core miner for **Pow Free Fair Token (PFFT)** on Ethereum mainnet.  
Automatically uses **all available CPU cores** on your VPS for maximum hashrate.

## Contract

`0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB`

- Max supply: 21,000,000 PFFT
- Per-wallet cap: 10,000 PFFT
- Free mint (0 ETH) — requires Proof-of-Work solve
- PoW difficulty increases as supply grows (24→40 bit)

## How it works

1. Reads current PoW challenge from contract
2. Splits the nonce search space evenly across **all CPU cores** in parallel
3. Each core brute-forces its slice (keccak256 hash must be below difficulty target)
4. Submits `freeMint(powNonce)` transaction (only costs gas)
5. Repeats until wallet cap (10,000 PFFT) or max supply reached

## Setup

```bash
# Clone
git clone https://github.com/Bores2511/pfftminer.git
cd pfftminer

# Install dependencies
pip install web3 pycryptodome

# Configure
cp .env.example .env
nano .env   # set PRIVATE_KEY=0x...

# Run — auto-detects ALL CPU cores
python3 pfft_miner.py
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ETH_RPC` | `https://ethereum-rpc.publicnode.com` | Ethereum RPC endpoint |
| `PRIVATE_KEY` | *(required)* | EVM wallet private key (with or without `0x`) |
| `NUM_WORKERS` | *(auto: all cores)* | Override number of CPU cores to use |

### Override core count (optional)

```bash
# Use only 4 cores out of 8
NUM_WORKERS=4 python3 pfft_miner.py

# Force all cores explicitly
NUM_WORKERS=8 python3 pfft_miner.py
```

## Run as systemd service (VPS / vastai)

```bash
# Copy service file
sudo cp pfft-miner.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now pfft-miner

# Check live logs
sudo journalctl -u pfft-miner -f
```

## Performance (Multi-Core vs Single-Core)

| Metric | Single-Core | 4-Core | 8-Core | 16-Core |
|--------|------------|--------|--------|---------|
| Hashrate | ~175k H/s | ~700k H/s | ~1.4M H/s | ~2.8M H/s |
| Time per mint (28-bit) | ~25 min | ~6 min | ~3 min | ~1.5 min |

*Actual performance depends on VPS CPU clock speed.*

## Current stats (May 2026)

| Metric | Value |
|--------|-------|
| Supply | ~6.5M / 21M (31%) |
| Difficulty | 28-bit (7 hex zeros) |
| PFFT per mint | ~284 |
| Gas per mint | ~0.00002 ETH |

## Security

- **NEVER commit your `.env` file** — it contains your private key
- `.env` is auto-added to `.gitignore`
- Use a **dedicated burner wallet** funded with only enough ETH for gas

## Files

| File | Description |
|------|-------------|
| `pfft_miner.py` | Main miner script (multi-core) |
| `.env.example` | Config template |
| `pfft-miner.service` | Systemd unit for VPS background running |
| `requirements.txt` | Python dependencies |

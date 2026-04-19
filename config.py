# ── Environment ───────────────────────────────────────────────────────────────
NUM_VMS       = 4
NUM_JOBS      = 1000
ARRIVAL_RATES = [10, 15, 20, 25, 30]   # jobs per second, swept in main.py

# ── SLA & Overload ─────────────────────────────────────────────────────────────
SLA_THRESHOLD      = 200   # ms — jobs exceeding this count as SLA violations
OVERLOAD_THRESHOLD = 0.80  # 80% CPU — paper's migration trigger point

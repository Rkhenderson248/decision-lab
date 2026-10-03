"""Precompute slow, deterministic results so pages open quickly after a cold start.

Run after changing lab/demand.py:  python scripts/precompute.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab import demand as dm  # noqa: E402

bt = dm.compute_backtest()
dm.BACKTEST_FILE.parent.mkdir(exist_ok=True)
bt.to_csv(dm.BACKTEST_FILE, index=False, float_format="%.6g")
print(f"wrote {dm.BACKTEST_FILE} ({len(bt):,} rows)")

"""Tiny tests you can run without the internet.

From the project folder, with your virtual environment on:

    python tests/test_portfolio.py

If every check prints OK, the return math matches a calculator.
"""

import sys
from pathlib import Path

# So this file can be run as: python tests/test_portfolio.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio import dividend_return, price_return, total_return


def check(name: str, actual, expected) -> None:
    # Round so tiny floating-point leftovers do not fail the test.
    if actual is None and expected is None:
        print(f"OK  {name}")
        return
    if actual is None or expected is None or round(actual, 10) != round(expected, 10):
        raise SystemExit(f"FAIL {name}: got {actual}, expected {expected}")
    print(f"OK  {name}")


def main() -> None:
    # Hand-checked: (110 - 100) / 100 = 0.10
    check("price up 10%", price_return(100, 110), 0.10)
    # (90 - 100) / 100 = -0.10
    check("price down 10%", price_return(100, 90), -0.10)
    # Missing start price -> no return yet (before Oct 20)
    check("no start price yet", price_return(None, 150), None)
    # $1 dividend on a $100 stock = 1%
    check("dividend adds 1%", dividend_return(100, 1.0), 0.01)
    # No dividend paid -> 0%
    check("no dividend", dividend_return(100, 0.0), 0.0)
    # (108 - 100 + 1) / 100 = 0.09
    check("total = price + dividend", total_return(100, 108, 1.0), 0.09)
    # A dividend can turn a small loss into a gain: (99 - 100 + 2) / 100 = 0.01
    check("dividend beats small drop", total_return(100, 99, 2.0), 0.01)
    # No current price -> no total yet
    check("no current price", total_return(100, None, 1.0), None)
    print("\nAll calculation checks passed.")


if __name__ == "__main__":
    main()

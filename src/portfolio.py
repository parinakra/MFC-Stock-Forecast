"""Return math for the competition.

This file does not talk to the internet. It only does arithmetic, which
makes it easy to test with a calculator.

Total return has two parts:
    price return    = (current price - start price) / start price
    dividend return = dividends paid during the contest / start price
    total return    = price return + dividend return

Only dividends whose ex-dividend date falls inside the contest count.
(We do NOT add the annual "dividend yield" - that is a whole year's worth.)
"""


def price_return(start_price: float | None, current_price: float | None) -> float | None:
    """Percent change in the share price.

    Example: start 100, current 110 -> (110 - 100) / 100 = 0.10 -> +10%
    """
    if start_price is None or current_price is None:
        return None
    if start_price <= 0:
        return None
    return (current_price - start_price) / start_price


def dividend_return(start_price: float | None, dividends: float | None) -> float | None:
    """Dividends paid per share during the contest, as a percent of the start price.

    Example: start 100, paid $1.00 -> 1 / 100 = 0.01 -> +1%
    """
    if start_price is None or start_price <= 0:
        return None
    return (dividends or 0.0) / start_price


def total_return(
    start_price: float | None, current_price: float | None, dividends: float | None
) -> float | None:
    """Price return + dividend return.

    Example: start 100, current 108, paid $1.00 -> (108 - 100 + 1) / 100 = 0.09 -> +9%
    """
    price = price_return(start_price, current_price)
    if price is None:
        return None
    return price + (dividend_return(start_price, dividends) or 0.0)


def format_price(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:,.2f}"


def format_return(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.2%}"

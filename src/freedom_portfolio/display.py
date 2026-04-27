"""
Terminal rendering helpers for the standalone CLI.

OPTIONAL — only loads when ``rich`` is installed.  The Telegram bot generates
PDFs through its own pipeline and never imports this module.
"""

from __future__ import annotations

from freedom_portfolio.models import Portfolio


def render_portfolio_table(portfolio: Portfolio):
    """Build a rich.Table summarising *portfolio*.  Returns the Table instance."""
    from rich.table import Table  # local import keeps rich optional

    pos_table = Table(title="Positions", show_lines=False)
    pos_table.add_column("Ticker",    style="cyan",   no_wrap=True)
    pos_table.add_column("Name",      style="white")
    pos_table.add_column("Qty",       justify="right")
    pos_table.add_column("Mkt Price", justify="right")
    pos_table.add_column("Value",     justify="right")
    pos_table.add_column("Cost",      justify="right")
    pos_table.add_column("P&L",       justify="right")
    pos_table.add_column("P&L %",     justify="right")

    for p in portfolio.pos:
        pnl     = p.profit_close if p.profit_close is not None else (p.s - p.open_bal)
        pnl_pct = (pnl / p.open_bal * 100) if p.open_bal else 0.0
        colour  = "green" if pnl >= 0 else "red"
        pos_table.add_row(
            p.i,
            p.name2 or p.name or "",
            f"{p.q:,.4f}".rstrip("0").rstrip("."),
            f"{p.mkt_price:,.2f}",
            f"{p.s:,.2f}",
            f"{p.open_bal:,.2f}",
            f"[{colour}]{pnl:+,.2f}[/{colour}]",
            f"[{colour}]{pnl_pct:+.2f}%[/{colour}]",
        )

    pos_table.add_section()
    pos_table.add_row(
        "TOTAL", "",
        "", "",
        f"{portfolio.total_position_value:,.2f}",
        "",
        f"{portfolio.total_pnl:+,.2f}",
        "",
    )
    return pos_table


def render_balance_table(portfolio: Portfolio):
    """Render account cash balances as a small rich.Table."""
    from rich.table import Table

    bal = Table(title="Balances")
    bal.add_column("Currency")
    bal.add_column("Free", justify="right")
    bal.add_column("FX → base", justify="right")
    for a in portfolio.acc:
        bal.add_row(a.curr, f"{a.s:,.2f}", f"{a.currval:.4f}")
    return bal


def print_portfolio(portfolio: Portfolio) -> None:
    """Print the full portfolio dashboard (balances + positions)."""
    from rich.console import Console

    console = Console()
    console.print(render_balance_table(portfolio))
    console.print()
    console.print(render_portfolio_table(portfolio))

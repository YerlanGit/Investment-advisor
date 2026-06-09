"""RAMP Telegram Bot — Python source root.

L-2: the former "Claude Code porting workspace" exports (commands, parity_audit,
port_manifest, query_engine, runtime, …) were removed.  `parity_audit` never
existed as a module, so importing it here broke `import src` and pytest
collection.  The bot runs with `PYTHONPATH=/app/src` and imports its modules
directly (`tg_bot`, `finance.*`, `freedom_portfolio.*`, …); it never relies on
the `src` package surface.
"""

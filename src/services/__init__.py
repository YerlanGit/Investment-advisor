"""Services package — runtime adapters for external systems (GCS, FRED, etc.).

Previously a placeholder for an archived subsystem; reactivated in Step 6.1
when report_storage.py + macro_data.py landed as production modules.
Kept minimal to avoid PYTHONPATH-dependent imports (Cloud Run sets
PYTHONPATH=/app/src, so `from src._archive_helper` would fail).
"""

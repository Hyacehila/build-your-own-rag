"""Preserve the existing page-audit module entry point."""

from .diagnostics.cleaning_audit import audit_pages as audit_pages
from .diagnostics.cleaning_audit import main

if __name__ == "__main__":
    main()

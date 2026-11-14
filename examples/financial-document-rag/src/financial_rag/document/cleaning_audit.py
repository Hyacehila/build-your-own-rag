"""Preserve the existing page-audit module entry point."""

from financial_rag.diagnostics.cleaning_audit import audit_pages as audit_pages
from financial_rag.diagnostics.cleaning_audit import main

if __name__ == "__main__":
    main()

"""Scheme CLIs delegate to the same frozen-plan executor as the serial CLI."""
import sys
from pathlib import Path


def main(arm):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'tools'))
    from contract_experiment import main as execute
    execute(arm=arm)

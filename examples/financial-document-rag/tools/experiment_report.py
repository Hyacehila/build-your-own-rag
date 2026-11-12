"""Common offline report entrypoint; optionally combine verified reused arms."""
import argparse
from pathlib import Path

from experiment_review.new_run import build_review

from financial_rag.common import read_json


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args(argv)
    build_review(args.plan,args.output)
    if read_json(args.plan).get('reuse_material'):
        from experiment_delivery import build_delivery
        build_delivery(args.plan)


if __name__ == "__main__":
    main()

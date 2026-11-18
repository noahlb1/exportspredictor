"""`python -m exportspredictor.loop <command>` (see `exportspredictor.loop.cli`)."""

import sys

from exportspredictor.loop.cli import main

main(sys.argv[1:])

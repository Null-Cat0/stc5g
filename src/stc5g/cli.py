"""Command-line entry point: ``stc5g <command> [options]``.

Each command forwards its options to the ``main(argv)`` function of one module, so
``stc5g train --epochs 5`` is equivalent to ``python -m stc5g.training.train --epochs 5``.
The Ryu applications are not commands: start them with ``ryu-manager`` (see the README).
"""

from __future__ import annotations

import importlib
import sys
from typing import Dict, List, Optional, Tuple

#: Command name -> (module, one-line description).
COMMANDS: Dict[str, Tuple[str, str]] = {
    "repair": ("stc5g.dataset.repair_pcaps", "Repair truncated/corrupt pcap files with editcap"),
    "durations": ("stc5g.dataset.pcap_durations", "Summarise capture durations per service"),
    "extract": ("stc5g.dataset.pcap_reader", "Convert pcap files into windowed throughput CSV files"),
    "label": ("stc5g.dataset.set_class", "Add the class column to the windowed CSV files"),
    "plot-dataset": ("stc5g.dataset.plot_throughput", "Plot throughput / time-delta figures of the dataset"),
    "pipeline-figure": ("stc5g.dataset.pipeline_figure", "Draw the data-pipeline figure of the paper"),
    "train": ("stc5g.training.train", "Train and evaluate the LSTM classifier"),
    "timeline": ("stc5g.viz.timeline", "Plot prediction timelines of a trained model"),
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = ["usage: stc5g <command> [options]", "", "commands:"]
    lines += [f"  {name:<{width}}  {desc}" for name, (_, desc) in COMMANDS.items()]
    lines += ["", "Run 'stc5g <command> --help' for the options of a command."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    """Dispatch ``argv[0]`` to the module that implements it."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(_usage())
        return
    command, rest = args[0], args[1:]
    if command not in COMMANDS:
        print(f"stc5g: unknown command '{command}'\n\n{_usage()}", file=sys.stderr)
        sys.exit(2)
    module = importlib.import_module(COMMANDS[command][0])
    sys.argv = [f"stc5g {command}", *rest]  # so that argparse prints the right program name
    module.main(rest)


if __name__ == "__main__":
    main()

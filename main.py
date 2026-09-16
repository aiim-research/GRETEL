"""GRETEL entry point.

    python main.py <config_file> [run_number]

The config file decides everything, including which evaluation manager runs
it. Two generations of configuration exist and they are told apart by their
top-level keys:

  * ``evaluator``      the current generation. The evaluation pipeline and its
                       stages are declared in that section
                       (``src.evaluation.future.evaluator.Evaluator``), and the
                       experiment is a list of ``doe-triplets``.
  * ``doe-triplets``   without ``evaluator``: the earlier triplet format, whose
                       metrics come from an ``evaluation_metrics`` list.
  * ``do-pairs``       the paired dataset/oracle format.
  * neither            the original flat format.

Picking the manager from the config rather than from the script name is what
lets one command run a configuration from any generation. ``future_main.py``
runs the current generation too, with OMP/MKL thread caps applied before torch
is imported; the cluster launchers call that one.
"""

import os
import sys

import torch  # noqa: F401  (imported for its side effects on BLAS threading)

from src.utils.context import Context

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _manager_for(conf):
    """Return the evaluation manager class this configuration asks for."""
    if "evaluator" in conf:
        from src.evaluation.future.evaluator_manager_triplets import EvaluatorManager
        return EvaluatorManager, "pipeline evaluator (current)"
    if "doe-triplets" in conf:
        from src.evaluation.evaluator_manager_triplets import EvaluatorManager
        return EvaluatorManager, "triplets, metric list"
    if "do-pairs" in conf:
        from src.evaluation.evaluator_manager_do import EvaluatorManager
        return EvaluatorManager, "do-pairs"
    from src.evaluation.evaluator_manager import EvaluatorManager
    return EvaluatorManager, "flat"


def main(argv):
    if len(argv) < 2:
        # Fall back to the environment so a container or a job script can set
        # the config without rewriting the command.
        if "GRETEL_CONFIG_FILE" not in os.environ:
            print("Usage: python main.py <config_file> [run_number]")
            return 1
        argv = argv + [os.environ["GRETEL_CONFIG_FILE"]]

    context = Context.get_context(argv[1])
    context.run_number = int(argv[2]) if len(argv) == 3 else -1

    manager_cls, flavour = _manager_for(context.conf)
    context.logger.info(f"Executing: {context.config_file} Run: {context.run_number}")
    context.logger.info(f"Creating the evaluation manager [{flavour}]...")
    eval_manager = manager_cls(context)

    context.logger.info("Evaluating the explainers...")
    eval_manager.evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

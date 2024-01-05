import os
import torch
#torch.manual_seed(5)#3,5
import random
#random.seed(0)
import numpy as np
#np.random.seed(0)

'''os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=1 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=1'''

from src.evaluation.evaluator_manager import EvaluatorManager
from src.evaluation.evaluator_manager_do import EvaluatorManager as PairedEvaluatorManager

from src.utils.context import Context
import sys

if __name__ == "__main__":
    print(f"Generating context for: {sys.argv[1]}")
    context = Context.get_context(sys.argv[1])
    context.run_number = int(sys.argv[2]) if len(sys.argv) == 3 else -1

    '''if torch.backends.mps.is_available():
        context.logger.info(f"MPS support founded switch to torch.set_default_dtype(torch.float32)")
        context.logger.info(f"Clean the cache if torch.float64 where used before")
        torch.set_default_dtype(torch.float32)'''

    context.logger.info(f"Executing: {context.config_file} Run: {context.run_number}")
    context.logger.info(
        "Creating the evaluation manager......................................................."
    )

    
    if 'do-pairs' in context.conf:
        context.logger.info("Creating the PAIRED evaluators...............................................................")
        eval_manager = PairedEvaluatorManager(context)
    else:
        context.logger.info("Creating the evaluators...............................................................")
        eval_manager = EvaluatorManager(context)

    context.logger.info(
        "Evaluating the explainers............................................................."
    )

    eval_manager.evaluate()

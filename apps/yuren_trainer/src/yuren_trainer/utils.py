"""
 Copyright 2023 Pleisto Inc

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """
import os
import time
from typing import Union

import torch
from deepspeed.runtime.engine import DeepSpeedEngine
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import logging


def create_rank_0_printer(rank: int, output_dir: str):
    """
    Creates a printer function that only prints & saves a message on the first process.

      Args:
        rank (int): The process rank.
        output_dir (str): The directory to save to.

      Returns:
        A printer function that only prints & saves a message on the first process.
    """

    # Create the output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        log_file = os.path.join(output_dir, f"print0_{int(time.time())}.txt")

        def print_once(msg: object):
            with open(log_file, "a") as f:
                print(msg)
                f.write(str(msg) + "\n")

        return print_once
    else:
        # If the rank is not 0, create a empty function that does nothing.
        def nothing(msg: object) -> None:
            return None

        return nothing


def create_logger(name: str, log_level: str, verbosity=False):
    """
    Creates a logger with the given name and log level.
    """
    logger = logging.get_logger(__name__)
    logger.setLevel(log_level)
    logging.set_verbosity(log_level)
    logging.enable_default_handler()
    logging.enable_explicit_format()
    if verbosity:
        logging.set_verbosity_info()
    return logger


def get_ds_state_dict(ds_engine: DeepSpeedEngine):
    """
    Get the deepspeed state dict.
    If it is zero stage 3, call all ranks regardless of the stage3_gather_16bit_weights_on_model_save parameter.
    """
    if ds_engine.zero_optimization_partition_weights():
        # consolidation is expensive in time and memory and therefore isn't a default
        state_dict = ds_engine._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = ds_engine.module.state_dict()
    return state_dict


def get_model_param_count(model: Union[DeepSpeedEngine, torch.nn.Module], trainable_only=False):
    """
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """
    if is_deepspeed_zero3_enabled() and isinstance(model, DeepSpeedEngine):

        def numel(p):
            return p.ds_numel

    else:

        def numel(p):
            return p.numel()

    return sum(numel(p) for p in model.parameters() if not trainable_only or p.requires_grad)

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

 Notes:

 This code is referenced from https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava.py
 Copyright 2023 Haotian Liu | Apache License 2.0
 """

from yuren_core.multimodal_llama import MultimodalLlama
from transformers import Trainer
from typing import Optional
import torch
import torch.nn as nn
import os

# TODO: wait for refactor and move from our internal monorepo to here,
# currently this is NOT actually used in the trainer.


import os
from typing import Optional
import torch
from torch import nn
from transformers import Trainer


class MultimodalLLamaTrainer(Trainer):
    """
    This is a subclass of the HuggingFace Trainer class, which is used to train Multimodal LLama models.
    """

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        Unwraps the model from the DataParallel wrapper if necessary.
        Args:
            model: A PyTorch nn.Module, possibly wrapped in a DataParallel wrapper.

        Returns:
            The unwrapped PyTorch nn.Module.
        """
        if hasattr(model, "module"):
            return self._unwrap_model(model.module)
        else:
            return model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            state_dict_to_save = state_dict
            if state_dict_to_save is None:
                # Only save the model itself if we are using distributed training
                model_to_save = self._unwrap_model(self.model)
                state_dict_to_save = model_to_save.state_dict()

            # Save the weights of specific layers
            weights_to_save = {}
            layer_keys_to_match = ["mm_projector", "embed_tokens", "embed_in"]
            for key, value in state_dict_to_save.items():
                if any(layer_key in key for layer_key in layer_keys_to_match):
                    weights_to_save[key] = value

            # Determine the output folder and save the weights
            current_folder = os.path.basename(output_dir)
            parent_folder = os.path.dirname(output_dir)

            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weights_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weights_to_save, os.path.join(output_dir, "mm_projector.bin")
                )

        # Call the parent class's _save method to handle the remaining save operations
        super(MultimodalLLamaTrainer, self)._save(output_dir, state_dict)

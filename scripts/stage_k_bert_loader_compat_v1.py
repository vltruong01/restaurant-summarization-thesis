"""Narrow deserialization compatibility for the SHA-pinned legacy RoBERTa binary.

Torch2.1 cannot meta-load this legacy pickle storage. The same weights-only CPU
load succeeds. Change storage placement only for this exact trusted file; never
switch checkpoint, dtype, layer, tokenizer or score implementation.
"""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from s1000_final_eval_core_v1 import file_hash,require
EXPECTED='36a10a8b694fadf9bf4f9049d14e257e88be45313ae02d882af9e60f39b8b2e8'
@contextmanager
def legacy_binary_cpu_load(path):
    import transformers.modeling_utils as module
    path=Path(path).resolve();require(file_hash(path)==EXPECTED,'Compatibility loader only permits exact pinned file')
    original=module.load_state_dict
    def load(checkpoint_file,*args,**kwargs):
        if Path(checkpoint_file).resolve()==path and kwargs.get('map_location')=='meta':kwargs['map_location']='cpu'
        return original(checkpoint_file,*args,**kwargs)
    with patch.object(module,'load_state_dict',load):yield

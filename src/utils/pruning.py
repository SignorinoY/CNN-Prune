from typing import Sequence, Tuple, Union

import torch.nn as nn
import torch.nn.utils.prune as pytorch_prune

_PARAM_TUPLE = Tuple[nn.Module, str]
_PARAM_LIST = Sequence[_PARAM_TUPLE]
_MODULE_CONTAINERS = (nn.Sequential, nn.ModuleList, nn.ModuleDict)


def filter_parameters_to_prune(module: nn.Module) -> _PARAM_LIST:
    modules = [m for m in module.modules() if not isinstance(m, _MODULE_CONTAINERS)]
    parameters = ("weight", "bias")
    parameters_to_prune = [
        (m, p) for p in parameters for m in modules if getattr(m, p, None) is not None
    ]
    _filtered_parameters_to_prune = []
    for module, name in parameters_to_prune:
        if isinstance(module, nn.Conv2d) and name == "weight":
            _filtered_parameters_to_prune.append((module, name))
    return _filtered_parameters_to_prune


def apply_pruning(
    parameters_to_prune: _PARAM_LIST, amount: float = 0.5, type: str = "global_unstructured"
):
    if type == "global_unstructured":
        pytorch_prune.global_unstructured(
            parameters_to_prune, pruning_method=pytorch_prune.L1Unstructured, amount=amount
        )
    elif type == "unstructured":
        for module, name in parameters_to_prune:
            pytorch_prune.l1_unstructured(module, name, amount=amount)
    elif type == "structured":
        for module, name in parameters_to_prune:
            pytorch_prune.ln_structured(module, name, amount=amount, dim=0, n=2)


def get_pruned_stats(module: nn.Module, name: str) -> Tuple[int, int]:
    """Get the number of zeros and total number of elements in a module."""
    attr = f"{name}_mask"
    if not hasattr(module, attr):
        return 0, 1
    mask = getattr(module, attr)
    return (mask == 0).sum().item(), mask.numel()


def log_sparsity_stats(parameters_to_prune: _PARAM_LIST):
    """Log sparsity stats for each parameter in the list."""
    total_params = sum(p.numel() for layer, _ in parameters_to_prune for p in layer.parameters())
    curr_stats = [get_pruned_stats(module, name) for module, name in parameters_to_prune]
    curr_total_zeros = sum(zeros for zeros, _ in curr_stats)
    print(f" {curr_total_zeros}/{total_params} ({curr_total_zeros / total_params:.2%})")
    for i, (module, name) in enumerate(parameters_to_prune):
        curr_mask_zeros, curr_mask_size = curr_stats[i]
        print(
            f"Pruned: {module!r}.{name}`"
            f" {curr_mask_zeros} ({curr_mask_zeros / curr_mask_size:.2%})"
        )


def remove_pruning(model: nn.Module):
    for _, module in model.named_modules():
        for k in list(module._forward_pre_hooks):
            hook = module._forward_pre_hooks[k]
            if isinstance(hook, pytorch_prune.BasePruningMethod):
                hook.remove(module)
                del module._forward_pre_hooks[k]

import pytest
import torch
import torch.nn as nn
from src.smr_adaptive_gating import adaptive_conformal_inference

def test_adaptive_conformal_inference_early_exit():
    torch.manual_seed(42)
    dim = 64
    num_tasks = 5
    num_classes = 10

    x = torch.randn(1, dim)
    prototypes = torch.randn(num_tasks, dim)
    precision_diag = torch.ones(dim)

    # Subcircuits with high-confidence prediction for candidate 0
    subcircuits = nn.ModuleList([
        nn.Sequential(nn.Linear(dim, num_classes)),
        nn.Sequential(nn.Linear(dim, num_classes))
    ])

    # Run inference with tau=0.0 (forcing early exit)
    res = adaptive_conformal_inference(x, None, prototypes[:2], precision_diag, subcircuits, tau_margin=0.0)
    assert res["early_exited"] is True
    assert res["candidates_evaluated"] == 1
    assert res["latency_multiplier"] == 1.0

def test_adaptive_conformal_inference_fallback():
    torch.manual_seed(42)
    dim = 64
    num_tasks = 2
    num_classes = 10

    x = torch.randn(1, dim)
    prototypes = torch.randn(num_tasks, dim)
    precision_diag = torch.ones(dim)

    subcircuits = nn.ModuleList([
        nn.Sequential(nn.Linear(dim, num_classes)),
        nn.Sequential(nn.Linear(dim, num_classes))
    ])

    # Run inference with unreachable high tau (forcing evaluation of both candidates)
    res = adaptive_conformal_inference(x, None, prototypes, precision_diag, subcircuits, tau_margin=100.0)
    assert res["early_exited"] is False
    assert res["candidates_evaluated"] == 2
    assert res["latency_multiplier"] == 2.0

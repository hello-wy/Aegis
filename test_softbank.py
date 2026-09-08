from types import SimpleNamespace

import pandas as pd
import torch

from dataset.gimo_dataset import parse_semantic_labels
from model.aegis import AEGIS, PoseResidualRefiner, SemanticPrototypeBank
from train import body_scene_contact_targets, expand_semantic_labels


def test_continuous_semantic_prior_and_zero_start_residuals():
    bank = SemanticPrototypeBank(4, num_classes=2, prototypes_per_class=3)
    feature = torch.randn(2, 5, 4)
    logits = torch.randn(2, 5, 2)
    assert bank(feature, logits).shape == (2, 5, 32)

    model = AEGIS.__new__(AEGIS)
    torch.nn.Module.__init__(model)
    model.prototype_bank = bank
    model.prototype_fusion = torch.nn.Linear(64, 32)
    model.prototype_gate = torch.nn.Parameter(torch.zeros(()))
    z_base = torch.randn(2, 5, 32)
    fused, _ = model._fuse_semantic_prior(feature, z_base, logits)
    assert torch.equal(fused, z_base)

    refiner = PoseResidualRefiner(feature_dim=4, scene_dim=3)
    delta = refiner(
        torch.randn(2, 5, 63), torch.randn(2, 5, 4), torch.randn(2, 5, 3),
        torch.randn(2, 5, 4), torch.randn(2, 5, 23),
    )
    assert torch.equal(delta, torch.zeros_like(delta))


def test_semantic_labels_and_contact_targets():
    labels = parse_semantic_labels(
        pd.Series({'token_motion_label': '0,1,2,3,4', 'motion_label': 0}), 5
    ).unsqueeze(0)
    assert expand_semantic_labels(labels, 2, 10).tolist() == [
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    ]

    joints = torch.tensor([[[[0.0, 0.0, 0.01], [0.0, 0.0, 1.0]]]])
    scene = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]])
    targets = body_scene_contact_targets(joints, scene, threshold=0.08, max_scene_points=2)
    assert targets.tolist() == [[[1.0, 0.0]]]

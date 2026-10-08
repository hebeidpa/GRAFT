import unittest
from types import SimpleNamespace

import torch

from graft_pillar.train_mask_guided import (
    binary_focal_loss_with_logits,
    objective_loss,
)


class FocalObjectiveTests(unittest.TestCase):
    def test_confident_correct_predictions_have_lower_focal_loss(self):
        targets = torch.tensor([1.0, 0.0])
        confident = binary_focal_loss_with_logits(
            torch.tensor([6.0, -6.0]), targets, alpha=0.25, gamma=2.0
        )
        uncertain = binary_focal_loss_with_logits(
            torch.tensor([0.0, 0.0]), targets, alpha=0.25, gamma=2.0
        )
        self.assertLess(float(confident), float(uncertain))

    def test_joint_objective_is_finite_and_differentiable(self):
        logits = torch.tensor([[0.2, -0.4], [0.6, 0.3]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        args = SimpleNamespace(
            focal_alpha=0.25,
            focal_gamma=2.0,
            lambda_recurrence=1.0,
            lambda_complication=0.75,
        )
        loss = objective_loss(logits, targets, [torch.tensor([1.0])], args)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from gptq import quantize


class GPTQTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.weight = torch.randn(12, 16)
        self.scales = self.weight.reshape(12, 4, 4).abs().amax(-1) / 7

    def test_identity_hessian_matches_independent_rounding(self):
        result, _ = quantize(self.weight, torch.eye(16), self.scales, 4, block_size=7)
        scale = self.scales.repeat_interleave(4, dim=1)
        expected = torch.clamp(torch.round(self.weight / scale), -8, 7) * scale
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_correlated_inputs_improve_reconstruction_and_preserve_original_groups(self):
        values = torch.randn(128, 16) @ torch.randn(16, 16)
        original = self.weight.clone()
        result, _ = quantize(self.weight, values.T @ values, self.scales, 4, block_size=5)
        scale = self.scales.repeat_interleave(4, dim=1)
        baseline = torch.round(self.weight / scale).clamp(-8, 7) * scale
        actual_loss = (values @ (result - self.weight).T).square().sum()
        baseline_loss = (values @ (baseline - self.weight).T).square().sum()
        self.assertLessEqual(float(actual_loss), float(baseline_loss) * 1.00001)
        self.assertLess(float(actual_loss), float(baseline_loss) * 0.99)
        torch.testing.assert_close(result, torch.round(result / scale).clamp(-8, 7) * scale, rtol=0, atol=0)
        torch.testing.assert_close(self.weight, original, rtol=0, atol=0)

    def test_invalid_hessian_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Hessian"):
            quantize(self.weight, torch.eye(15), self.scales, 4)


if __name__ == "__main__":
    unittest.main()

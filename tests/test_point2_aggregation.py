import unittest

import torch

from point2.runner import interpolate, weighted_mean


class Point2AggregationTests(unittest.TestCase):
    def test_weighted_aggregation_preserves_both_factors(self):
        updates = [
            {"example_count": 1, "adapter": {"lora_A": torch.tensor([1.0]), "lora_B": torch.tensor([2.0])}},
            {"example_count": 3, "adapter": {"lora_A": torch.tensor([5.0]), "lora_B": torch.tensor([6.0])}},
        ]
        result = weighted_mean(updates)
        self.assertAlmostEqual(result["lora_A"].item(), 4.0)
        self.assertAlmostEqual(result["lora_B"].item(), 5.0)

    def test_async_interpolation_does_not_mutate_inputs(self):
        current = {"lora_A": torch.tensor([0.0]), "lora_B": torch.tensor([2.0])}
        incoming = {"lora_A": torch.tensor([4.0]), "lora_B": torch.tensor([6.0])}
        result = interpolate(current, incoming, 0.25)
        self.assertEqual(result["lora_A"].item(), 1.0)
        self.assertEqual(result["lora_B"].item(), 3.0)
        self.assertEqual(current["lora_A"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()

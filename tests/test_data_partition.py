import unittest
from unittest.mock import patch

from dataset.utils import split_dir


class DirichletSplitTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {'input_ids': f'example-{i}', 'label': str(i), 'category': f'topic-{i % 4}'}
            for i in range(120)
        ]

    def _split(self):
        outputs = {}
        config = {
            'client_num': 4,
            'train_ratio': 0.8,
            'alpha': 0.5,
            'seed': 42,
            'min_samples_per_client': 5,
            'max_split_attempts': 1000,
            'dir_path': 'partition',
        }

        def capture(rows, path):
            outputs[path] = [row['input_ids'] for row in rows]

        with patch('dataset.utils.os.makedirs'), patch('dataset.utils.save_file', side_effect=capture):
            split_dir(self.records, config)
        return outputs

    def test_split_is_complete_disjoint_and_reproducible(self):
        parts = self._split()
        self.assertEqual(parts, self._split())
        self.assertEqual(len(parts), 8)
        self.assertTrue(all(parts.values()))
        assigned = [item for rows in parts.values() for item in rows]
        self.assertEqual(len(assigned), len(self.records))
        self.assertEqual(set(assigned), {row['input_ids'] for row in self.records})

    def test_impossible_minimum_fails_instead_of_looping_forever(self):
        with self.assertRaisesRegex(ValueError, 'Not enough records'):
            split_dir(self.records[:10], {
                'client_num': 4,
                'train_ratio': 0.8,
                'alpha': 0.5,
                'dir_path': 'partition',
                'min_samples_per_client': 5,
            })


if __name__ == '__main__':
    unittest.main()

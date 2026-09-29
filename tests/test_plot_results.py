"""Data-integrity checks for optional paper-figure generation.

Run after installing requirements-plot.txt:
    python -m unittest discover -s tests -p test_plot_results.py
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.plot_results import collect_summary


class PlotDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / 'data/our_cost/F'
        self.directory.mkdir(parents=True)
        inputs = self.root / 'data/test_cases'
        inputs.mkdir()
        # These keys differ beyond floating-point precision and exceed int64.
        self.functions = [str((1 << 2000) + i) for i in range(4)]
        (inputs / 'test_cases_11').write_text('\n'.join(self.functions) + '\n')
        for metric, values in [('t', [10, 20]), ('cx', [3, 4])]:
            (self.directory / f'our_{metric}_cost_F_11_1_3.txt').write_text(
                ''.join(f'{key}:{value}\n' for key, value in zip(self.functions[1:3], values)))
        self.report = self.directory / 'our_results_F_11_1_3.txt'
        self.report.write_text(
            'number of input variables: 11 (2 cases); path=F\n'
            'avg synthesize wall: 1.25 ms/case\n')

    def collect(self):
        return collect_summary(['F'], [11], 1, 3, root=self.root)

    def test_large_keys_and_nonzero_interval(self):
        # A saved interval must not be sliced a second time.
        row = self.collect().iloc[0]
        self.assertEqual((row['total_t'], row['total_cx'], row['cases']), (30, 7, 2))
        self.assertEqual(row['avg_wall_ms'], 1.25)
        self.assertEqual(row['timing_source'], 'data/our_cost/F/our_results_F_11_1_3.txt')

    def test_rejects_misaligned_keys(self):
        path = self.directory / 'our_t_cost_F_11_1_3.txt'
        path.write_text(''.join(f'{key}:10\n' for key in reversed(self.functions[1:3])))
        with self.assertRaisesRegex(ValueError, 'Case count/order'):
            self.collect()

    def test_larger_batch_time_cannot_replace_exact_batch(self):
        self.report.rename(self.directory / 'our_results_F_11_0_4.txt')
        with self.assertRaisesRegex(FileNotFoundError, 'Missing exact batch'):
            self.collect()

    def test_rejects_wrong_report_route_or_count(self):
        for header in ('11 (4 cases); path=F', '11 (2 cases); path=FS'):
            with self.subTest(header=header):
                self.report.write_text(f'number of input variables: {header}\n'
                                       'avg synthesize wall: 1.25 ms/case\n')
                with self.assertRaisesRegex(ValueError, 'Run header'):
                    self.collect()


if __name__ == '__main__':
    unittest.main()

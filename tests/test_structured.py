"""Source extraction, oracle correctness, and experiment-recording regressions."""
import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from experiments.generate_structured_cases import (
    DATASETS, DATA_DIR, component_tables, generate, load_cases,
)
from evaluate_structured import (
    FIELDS, PROTOCOL, complete_means, cost_and_verify, main, read_records,
    run_case, save_records, summarize, tt_words,
)


INFO = {'implementation': 'test-backend', 'environment': 'test-environment'}


def slow_worker(connection, method, tt, n, options):
    connection.send(('ready', INFO))
    connection.recv()
    time.sleep(10)


def broken_worker(connection, method, tt, n, options):
    connection.send(('ready', INFO))
    connection.recv()
    os._exit(7)


def record(index, tt, method='F', status='complete'):
    row = dict.fromkeys(FIELDS, '')
    row.update(dataset='des', n=6, index=index, truth_table=tt, method=method,
               protocol=PROTOCOL, status=status, **INFO)
    if status == 'complete':
        row.update(t_count=7, cx_count=6, synthesis_ms=1.0, verification='exhaustive-X-permutation')
    return row


class StructuredCasesTests(unittest.TestCase):
    def test_checked_in_sources_and_widths(self):
        for name, spec in DATASETS.items():
            with self.subTest(name=name):
                cases = load_cases(name)
                self.assertEqual(len(cases), spec.count)
                self.assertTrue(all(0 <= tt < 1 << (1 << spec.n) for tt in cases))
        self.assertEqual(set(DATASETS), {'des', 'aes', 'ascon', 'speedy'})
        with (DATA_DIR / 'case_index.csv').open() as handle:
            mapping = list(csv.DictReader(handle))
        self.assertEqual(len(mapping), 51)
        self.assertEqual(mapping[0]['output'], 'S1[0]')
        self.assertEqual(mapping[31]['output'], 'S8[3]')
        for start, width in ((32, 8), (40, 5), (45, 6)):
            self.assertEqual([row['output'] for row in mapping[start:start + width]],
                             [f'Sbox[{j}]' for j in range(width)])
        self.assertEqual([row['dataset'] for row in mapping],
                         ['des'] * 32 + ['aes'] * 8 + ['ascon'] * 5 + ['speedy'] * 6)

    def test_extraction_bit_order_and_constants(self):
        self.assertEqual(component_tables([3, 0, 2, 1], 2), [9, 5])
        for name, assignments in [('aes', {0: 0x63, 0x53: 0xed, 255: 0x16}),
                                  ('des', {0b011011: 5}),
                                  ('ascon', {0: 4, 1: 0x0b, 0x10: 0x1e, 0x1f: 0x17}),
                                  ('speedy', {0: 8, 1: 0, 0x20: 2, 0x3f: 0x3d})]:
            tables, _, _ = generate(name)
            width = 4 if name == 'des' else DATASETS[name].count
            for assignment, expected in assignments.items():
                actual = sum(((tables[j] >> assignment) & 1) << j for j in range(width))
                self.assertEqual(actual, expected)

    def test_reject_corrupt_cases(self):
        with TemporaryDirectory() as tmp:
            spec = DATASETS['des']
            cases = load_cases('des')
            cases[0], cases[1] = cases[1], cases[0]
            (Path(tmp) / spec.filename('des')).write_text(''.join(f'{tt}\n' for tt in cases))
            with self.assertRaisesRegex(ValueError, 'pinned sources'):
                load_cases('des', tmp)

    def test_truth_table_word_padding(self):
        self.assertEqual(tt_words(0, 8), [0, 0, 0, 0])
        self.assertEqual(tt_words(1, 10), [1] + [0] * 15)
        self.assertEqual(tt_words((1 << 255) | 3, 8), [3, 0, 0, 1 << 63])
        self.assertEqual(tt_words(1, 6), [1])


class OracleVerificationTests(unittest.TestCase):
    def test_input_order_open_controls_and_cost(self):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import MCXGate
        circuit = QuantumCircuit(4)
        circuit.append(MCXGate(3, ctrl_state=1), range(4))
        self.assertEqual(cost_and_verify(circuit, 1 << 1, 3), (15, 12))
        self.assertEqual(cost_and_verify(circuit, 1 << 4, 3, reversed_inputs=True), (15, 12))
        with self.assertRaises(ValueError):
            cost_and_verify(circuit, 1 << 1, 3, reversed_inputs=True)

    def test_reject_input_damage_and_hidden_phase(self):
        from qiskit import QuantumCircuit
        circuit = QuantumCircuit(4)
        circuit.x(0)
        with self.assertRaises(ValueError):
            cost_and_verify(circuit, 0, 3)
        circuit = QuantumCircuit(4)
        circuit.z(0)
        with self.assertRaisesRegex(ValueError, 'Unsupported gate'):
            cost_and_verify(circuit, 0, 3)
        circuit = QuantumCircuit(4)
        circuit.global_phase = 0.1
        with self.assertRaises(ValueError):
            cost_and_verify(circuit, 0, 3)


class StructuredRecordingTests(unittest.TestCase):
    def test_timeout_is_not_a_measurement_and_worker_is_killed(self):
        result, info = run_case('sshr', 0, 6, {}, 0.1, worker=slow_worker)
        self.assertEqual(result['status'], 'timeout')
        self.assertNotIn('t_count', result)
        self.assertNotIn('synthesis_ms', result)
        self.assertEqual(info, INFO)

    def test_crashed_backend_is_an_error(self):
        result, _ = run_case('sshr', 0, 6, {}, 5, worker=broken_worker)
        self.assertEqual(result['status'], 'error')
        self.assertIn('exited unexpectedly', result['error'])

    def test_resume_guards_alignment_and_incomplete_summary(self):
        cases = load_cases('des')
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'des/F'
            directory.mkdir(parents=True)
            path = directory / 'results.csv'
            save_records(path, [record(0, cases[0])])
            before = path.read_bytes()
            args = ['run', 'des', 'F', '--stop', '1', '--output-dir', tmp]
            with self.assertRaises(FileExistsError):
                main(args)
            with patch('evaluate_structured.run_case', side_effect=AssertionError('reran a completed case')):
                main(args + ['--resume'])
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((directory / '.running').exists())
            report = summarize('des', tmp)
            self.assertIn('complete=1/32', report)
            self.assertNotIn('mean t_count=', report)
            bad = record(0, cases[1])
            save_records(path, [bad])
            with self.assertRaisesRegex(ValueError, 'Misaligned'):
                read_records(path, 'des', 'F', cases)

    def test_partial_baseline_cannot_produce_reductions(self):
        cases = load_cases('des')
        with TemporaryDirectory() as tmp:
            for method in ('F', 'exorcism', 'sshr'):
                directory = Path(tmp) / 'des' / method
                directory.mkdir(parents=True)
                rows = [record(i, tt, method) for i, tt in enumerate(cases)]
                if method == 'sshr':
                    rows[0] = record(0, cases[0], method, 'timeout')
                save_records(directory / 'results.csv', rows)
            report = summarize('des', tmp)
            self.assertIn('F vs exorcism: t_count reduction=0.000%', report)
            self.assertNotIn('F vs sshr', report)
            self.assertIn('sshr: complete=31/32, timeout=1', report)
            self.assertIsNone(complete_means(rows, 32))


if __name__ == '__main__':
    unittest.main()

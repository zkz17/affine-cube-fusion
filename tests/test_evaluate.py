"""Exercise real evaluation routes and their output layout in an isolated data directory."""
from contextlib import chdir, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from backend.OracleSynthesizer import OracleSynthesizer
from data import compare_no_m2
from evaluate import COVER_ROUTES, RAW_ROUTES, ROUTES, count_costs, evaluate_all, load_aligned, main as evaluate_main
from experiments import data_summary
from utils.TruthTable import TruthTable
from utils.to_qiskit import to_qiskit


def reference_costs(route, functions, m2):
    """Explicit zero-ancilla synthesis is the reference for the public evaluator."""
    s = OracleSynthesizer(ancilla_num=0, m2_post=m2, gl_search=route == 'FG',
                         fprm_proxy=route == 'FP',
                         paths=(route,) if route in ('A', 'F') else ('FS',))
    result = []
    for fn in functions:
        tt = TruthTable(fn, 4)
        if route in COVER_ROUTES:
            s.clear()
            s.allocate_input_qubit(4)
            s.process_cover_route(tt, route)
        else:
            s.synthesize(tt)
        assert s.circ.verify(tt), (route, fn)
        assert not s.qubit_mgr.anc, (route, fn)
        result.append(count_costs(to_qiskit(s.circ)))
    return result


def test_evaluation_routes(verbose=False):
    functions = [1, 278, 64560]
    print('Testing evaluation routes, verification and M2 output layout...')
    with TemporaryDirectory() as tmp, chdir(tmp):
        cases = Path('data/test_cases')
        cases.mkdir(parents=True)
        (cases / 'test_cases_4').write_text(''.join(f'{fn}\n' for fn in functions))
        for route in ROUTES:
            for m2 in (True, False):
                expected = reference_costs(route, functions, m2)
                with redirect_stdout(StringIO()):
                    evaluate_all(4, 0, len(functions), route, m2_post=m2,
                                 overwrite=route in RAW_ROUTES and not m2)
                tag = route if m2 or route in RAW_ROUTES else route + '_noM2'
                directory = Path('data/our_cost') / route
                for index, metric in enumerate(('t', 'cx')):
                    output = directory / f'our_{metric}_cost_{tag}_4_0_3.txt'
                    rows = [tuple(map(int, line.split(':')))
                            for line in output.read_text().splitlines()]
                    assert rows == [(fn, costs[index])
                                    for fn, costs in zip(functions, expected)], (route, m2, metric)
                report = directory / f'our_results_{tag}_4_0_3.txt'
                assert f'path={tag}' in report.read_text()
            if route in RAW_ROUTES:
                assert not list(directory.glob('*_noM2_*')), (route, 'raw route got noM2 files')
        assert {p.name for p in Path('data/our_cost').iterdir()} == set(ROUTES)

        # The existing consumer must be able to read freshly produced no-M2 files.
        with patch.object(compare_no_m2, 'OUR_COST', 'data/our_cost'), \
                patch.object(compare_no_m2, 'OUT', 'data/our_cost/no_M2_vs_default.txt'), \
                redirect_stdout(StringIO()):
            compare_no_m2.main()
        report = Path('data/our_cost/no_M2_vs_default.txt').read_text()
        for route in set(ROUTES) - set(RAW_ROUTES):
            assert f'****** {route} n=4 (3 cases) ******' in report
        routes = sorted(set(ROUTES) - set(RAW_ROUTES))
        positions = [report.index(f'****** {route} n=4') for route in routes]
        assert positions == sorted(positions), 'report route order depends on path separators'

        # A bad circuit must fail evaluation before its cost is accepted.
        with patch.object(OracleSynthesizer, 'synthesize', autospec=True,
                          side_effect=lambda self, tt: self.clear()), redirect_stdout(StringIO()):
            try:
                evaluate_all(4, 0, 1, 'FS')
            except AssertionError as exc:
                assert 'circuit verification failed' in str(exc)
            else:
                raise AssertionError('evaluation accepted an incorrect circuit')
    print('Evaluation integration test passed!!!')


def test_evaluation_protection():
    """CLI reruns cannot silently replace results, and failed runs preserve old data."""
    check = TestCase()
    with TemporaryDirectory() as tmp, chdir(tmp), redirect_stdout(StringIO()):
        cases = Path('data/test_cases/test_cases_4')
        cases.parent.mkdir(parents=True)
        cases.write_text('1\n278\n64560\n')
        args = ['4', '1', '3', 'F', '--output-dir', 'rerun results']
        evaluate_main(args)
        directory = Path('rerun results/F')
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        assert len(before) == 3 and not Path('data/our_cost').exists()
        rows = (directory / 'our_t_cost_F_4_1_3.txt').read_text().splitlines()
        assert [int(row.split(':')[0]) for row in rows] == [278, 64560]

        with patch.object(OracleSynthesizer, 'synthesize') as synthesize:
            with check.assertRaisesRegex(FileExistsError, 'Refusing to overwrite'):
                evaluate_all(4, 1, 3, 'F', output_dir='rerun results')
            synthesize.assert_not_called()

        original = OracleSynthesizer.synthesize

        def fail_second(synth, tt):
            if tt.bitstr == 64560:
                synth.clear()
            else:
                original(synth, tt)

        with patch.object(OracleSynthesizer, 'synthesize', fail_second):
            with check.assertRaisesRegex(AssertionError, 'circuit verification failed'):
                evaluate_all(4, 1, 3, 'F', output_dir='rerun results', overwrite=True)
            with check.assertRaisesRegex(AssertionError, 'circuit verification failed'):
                evaluate_all(4, 1, 3, 'F', output_dir='failed run')
        assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
        assert not Path('failed run').exists()

        evaluate_main(args + ['--overwrite'])
        for name in before:
            if name.startswith(('our_t_cost_', 'our_cx_cost_')):
                assert (directory / name).read_bytes() == before[name]
        for start, stop in [(-1, 2), (1, 1), (2, 1), (0, 4)]:
            with check.assertRaisesRegex(ValueError, 'input cases'):
                evaluate_all(4, start, stop, 'F', output_dir='invalid range')
        assert not Path('invalid range').exists()

        # Even a lone old report must protect the entire batch.
        for p in directory.glob('our_*_cost_*'):
            p.unlink()
        with check.assertRaises(FileExistsError):
            evaluate_all(4, 1, 3, 'F', output_dir='rerun results')

        # Simulate another writer winning after synthesis but during publication.
        original_open = Path.open
        collision = Path('concurrent/F/our_cx_cost_F_4_1_3.txt')

        def concurrent_open(target, mode='r', *args, **kwargs):
            if mode == 'x' and target == collision:
                with original_open(target, 'w') as stream:
                    stream.write('other writer\n')
            return original_open(target, mode, *args, **kwargs)

        with patch.object(Path, 'open', concurrent_open), check.assertRaises(FileExistsError):
            evaluate_all(4, 1, 3, 'F', output_dir='concurrent')
        assert collision.read_text() == 'other writer\n'
        assert list(collision.parent.iterdir()) == [collision]
    print('Evaluation output protection tests passed!!!')


def test_baseline_alignment():
    """Reject shifted or malformed baselines instead of silently comparing different functions."""
    check = TestCase()
    with TemporaryDirectory() as tmp:
        cases = Path(tmp) / 'cases'
        baseline = Path(tmp) / 'baseline'
        cases.write_text('1\n278\n64560\n')
        assert load_aligned(cases, baseline, 1, 3) == ([278, 64560], None)
        baseline.write_text('')
        assert load_aligned(cases, baseline, 1, 3) == ([278, 64560], None)
        baseline.write_text('1:0\n278:7\n64560:0\n')
        assert load_aligned(cases, baseline, 1, 3) == ([278, 64560], [7, 0])
        baseline.write_text('1:0\n278:7\n')
        assert load_aligned(cases, baseline, 1, 3) == ([278, 64560], None)
        for row, message in [('64560:7', 'truth-table mismatch'),
                             ('278:-1', 'Negative baseline'), ('invalid', 'Malformed baseline')]:
            baseline.write_text(f'1:0\n{row}\n')
            with check.assertRaisesRegex(ValueError, message):
                load_aligned(cases, baseline, 1, 3)
    print('Baseline alignment tests passed!!!')


def test_default_synthesis():
    """A constructor without options must use the same zero-ancilla FS route as evaluation."""
    default = OracleSynthesizer()
    reference = OracleSynthesizer(ancilla_num=0, paths=('FS',), m2_post=True)
    with patch.object(default, 'construct', side_effect=AssertionError('used legacy backend')):
        for fn in (0, 1, 278, 64560, 65535):
            tt = TruthTable(fn, 4)
            default.synthesize(tt)
            reference.synthesize(tt)
            assert default.circ.verify(tt) and not default.qubit_mgr.anc
            assert count_costs(to_qiskit(default.circ)) == count_costs(to_qiskit(reference.circ))
    print('Default zero-ancilla synthesis tests passed!!!')


def test_summary_ranges():
    """Consumers match separate batches and use the correct nonzero input interval."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        route = root / 'our_cost' / 'F'
        route.mkdir(parents=True)
        (root / 'test_cases').mkdir()
        (root / 'exorcism_cost').mkdir()
        functions = [1, 278, 64560, 444]
        (root / 'test_cases/test_cases_4').write_text(''.join(f'{fn}\n' for fn in functions))
        (root / 'exorcism_cost/exorcism_t_cost_4').write_text(
            ''.join(f'{fn}:{cost}\n' for fn, cost in zip(functions, [999, 40, 40, 60])))

        for start, stop, tc, cx in [(1, 2, [20], [10]), (2, 4, [30, 50], [15, 25])]:
            for tag, extra in [('F', 0), ('F_noM2', 10)]:
                for metric, costs in [('t', tc), ('cx', cx)]:
                    (route / f'our_{metric}_cost_{tag}_4_{start}_{stop}.txt').write_text(
                        ''.join(f'{fn}:{cost + extra}\n'
                                for fn, cost in zip(functions[start:stop], costs)))
                (route / f'our_results_{tag}_4_{start}_{stop}.txt').write_text(
                    'avg synthesize wall: 0.15 ms/case\n')

        output = root / 'comparison.txt'
        with patch.object(compare_no_m2, 'OUR_COST', str(root / 'our_cost')), \
                patch.object(compare_no_m2, 'OUT', str(output)), redirect_stdout(StringIO()):
            compare_no_m2.main()
        report = output.read_text(encoding='utf-8')
        for expected in ('baseline 30, ours 20', 'baseline 20, ours 10',
                         'baseline 100, ours 80', 'baseline 60, ours 40'):
            assert expected in report, expected

        output = root / 'new_directory/summary.txt'
        with patch.object(data_summary, 'OUR', str(root / 'our_cost')), \
                patch.object(data_summary, 'BASE', str(root)), \
                patch.object(data_summary, 'OUT', str(output)), redirect_stdout(StringIO()):
            data_summary.main()
        report = output.read_text(encoding='utf-8')
        assert '[2,4)' in report and '0.15' in report
        totals = report.split('=== ΣT ===')[1].split('=== ΣCX ===')[0].splitlines()[-2].split()
        assert totals == ['4', '80', '100'], totals
        assert '+20.0%' in report and '+0.0%' in report

        # Overlapping measurements may be combined only if they agree.
        (route / 'our_t_cost_F_4_0_2.txt').write_text('1:0\n278:21\n')
        try:
            compare_no_m2.load_default_map(str(route / 'our_t_cost_F_4_*.txt'))
        except AssertionError as exc:
            assert 'conflicting default cost' in str(exc)
        else:
            raise AssertionError('conflicting measurements were silently accepted')
    print('Data summary interval tests passed!!!')


def main(verbose=False):
    test_evaluation_routes(verbose)
    test_evaluation_protection()
    test_baseline_alignment()
    test_default_synthesis()
    test_summary_ranges()


if __name__ == '__main__':
    main()

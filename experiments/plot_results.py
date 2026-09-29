"""Plot matched synthesis batches as publication figures.

Run from the repository root:
    ./venv/bin/python -m experiments.plot_results
    ./venv/bin/python -m experiments.plot_results --metric cx
    ./venv/bin/python -m experiments.plot_results --metric time
    ./venv/bin/python -m experiments.plot_results --metric time --time-break 13000 27000

Cost figures compare eight optimized configurations with absolute and F-normalized
panels. Time figures show the same eight optimized configurations by default;
raw-cover baselines remain available for separate comparisons via --routes.
DD0/DD are display names for the saved A0/A routes; source names remain unchanged.
All figures use cases [0, 1000) for n=5,...,11 by default.
Only exact run intervals are accepted: a larger batch's runtime cannot be sliced.
Writes a reusable T/CX/time summary CSV and PDF/PNG figures under figure/results/.
"""
import argparse
from itertools import islice
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use('Agg')  # Save figures in WSL without requiring a GUI.
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator, ScalarFormatter, StrMethodFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ('MT0', 'MT', 'PK0', 'PK', 'PKS', 'PKP', 'F0', 'F', 'FS', 'FP', 'A0', 'A')
OPTIMIZED_ROUTES = ('MT', 'PK', 'PKS', 'PKP', 'F', 'FS', 'FP', 'A')
FUSION_PAIRS = (('MT0', 'MT'), ('PK0', 'PK'), ('F0', 'F'), ('A0', 'A'))
ROUTE_LABELS = {route: {'A0': 'DD0', 'A': 'DD'}.get(route, route) for route in ROUTES}

# Color identifies the cover family; line/marker identify the optimization stage.
FAMILY_COLORS = {'MT': '#D55E00', 'PK': '#009E73', 'F': '#0072B2', 'DD': '#D4AA00'}
ROUTE_STYLES = {
    'MT0': ('MT', 'raw'), 'MT': ('MT', 'fusion'),
    'PK0': ('PK', 'raw'), 'PK': ('PK', 'fusion'),
    'PKS': ('PK', 'shift'), 'PKP': ('PK', 'proxy'),
    'F0': ('F', 'raw'), 'F': ('F', 'fusion'),
    'FS': ('F', 'shift'), 'FP': ('F', 'proxy'),
    'A0': ('DD', 'raw'), 'A': ('DD', 'fusion'),
}
STAGE_STYLES = {
    'raw': (':', 's'), 'fusion': ('--', 'o'),
    'shift': ('-', '^'), 'proxy': ('-.', 'D'),
}
# Show dash-dot cycles outside the central diamond marker in every legend.
LEGEND_HANDLE_LENGTH = 4.0
METRICS = {
    't': ('total_t', 'Total T cost'),
    'cx': ('total_cx', 'Total CX cost'),
    'time': ('avg_wall_ms', 'Mean synthesis time (ms/function)'),
}
PLOT_STYLE = {
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white',
}


def load_costs(path, functions):
    """Keep truth-table keys as strings: n=11 keys exceed machine integers."""
    frame = pd.read_csv(path, sep=':', header=None, names=['tt', 'cost'],
                        dtype={'tt': 'string', 'cost': 'int64'},
                        keep_default_na=False)
    if frame['tt'].tolist() != functions:
        raise ValueError(f'Case count/order does not match the requested inputs: {path}')
    if (frame['cost'] < 0).any():
        raise ValueError(f'Negative cost in {path}')
    return int(frame['cost'].sum())


def collect_summary(routes=ROUTES, sizes=range(5, 12), start=0, stop=1000, root=ROOT):
    """Read exact batches and validate their inputs, counts, labels, and timings."""
    if not 0 <= start < stop:
        raise ValueError('Require 0 <= start < stop.')
    rows = []
    for n in sizes:
        with (root / 'data/test_cases' / f'test_cases_{n}').open() as stream:
            functions = [line.strip() for line in islice(stream, start, stop)]
        if len(functions) != stop - start:
            raise ValueError(f'Insufficient test cases: n={n}, [{start}, {stop})')
        for route in routes:
            directory = root / 'data/our_cost' / route
            suffix = f'{route}_{n}_{start}_{stop}.txt'
            t_path = directory / f'our_t_cost_{suffix}'
            cx_path = directory / f'our_cx_cost_{suffix}'
            report_path = directory / f'our_results_{suffix}'
            missing = [str(p) for p in (t_path, cx_path, report_path) if not p.is_file()]
            if missing:
                raise FileNotFoundError(
                    f'Missing exact batch: {missing}. Generate it with '
                    f'python evaluate.py {n} {start} {stop} {route}')
            # Some historical reports contain non-UTF-8 prose; numeric fields are ASCII.
            report = report_path.read_text(encoding='utf-8', errors='replace')
            header = re.search(
                r'number of input variables: (\d+) \((\d+) cases\); paths?=(\w+)',
                report)
            if not header or header.groups() != (str(n), str(stop - start), route):
                raise ValueError(f'Run header does not match the requested batch: {report_path}')
            timing = re.search(r'avg synthesize wall: ([\d.]+) ms', report)
            if not timing:
                raise ValueError(f'Missing synthesis time: {report_path}')
            rows.append({
                'route': route, 'display_name': ROUTE_LABELS[route],
                'n': n, 'start': start, 'stop': stop,
                'cases': stop - start,
                'total_t': load_costs(t_path, functions),
                'total_cx': load_costs(cx_path, functions),
                'avg_wall_ms': float(timing[1]),
                't_source': t_path.relative_to(root).as_posix(),
                'cx_source': cx_path.relative_to(root).as_posix(),
                'timing_source': report_path.relative_to(root).as_posix(),
            })
    return pd.DataFrame(rows)


def draw_figure(summary, routes, metric, scale, output_dir):
    summary = summary[summary['route'].isin(routes)]
    column, ylabel = METRICS[metric]
    sizes = sorted(summary['n'].unique())
    cases = int(summary['cases'].iloc[0])
    if scale == 'log' and (summary[column] <= 0).any():
        raise ValueError(f'Log scale requires strictly positive {column} values.')

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.9))
        # Fixed margins keep the two exports the same size and leave room for the key.
        fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.23)
        for route in routes:
            data = summary[summary['route'] == route].sort_values('n')
            family, stage = ROUTE_STYLES[route]
            linestyle, marker = STAGE_STYLES[stage]
            ax.plot(data['n'], data[column], label=ROUTE_LABELS[route],
                    color=FAMILY_COLORS[family], linestyle=linestyle,
                    marker=marker, markersize=5.2, markerfacecolor='white',
                    markeredgewidth=1.1, linewidth=1.6,
                    zorder=3 if stage in ('shift', 'proxy') else 2)
        ax.set_xticks(sizes)
        ax.set_xlim(min(sizes) - 0.12, max(sizes) + 0.12)
        ax.set_xlabel('Number of input variables, $n$')
        ax.set_ylabel(ylabel + (' (log scale)' if scale == 'log' else ''))
        ax.set_yscale(scale)
        if scale == 'linear':
            ax.set_ylim(bottom=0)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)
        else:
            ax.margins(y=0.06)
        ax.grid(axis='y', which='major', color='#D8D8D8', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f'{cases:,} matched functions per input size', pad=12)
        fig.legend(*ax.get_legend_handles_labels(), loc='lower center',
                   bbox_to_anchor=(0.55, 0.035), ncol=4 if len(routes) == 8 else min(6, len(routes)),
                   frameon=False, fontsize=9.5, handlelength=LEGEND_HANDLE_LENGTH,
                   columnspacing=1.1, labelspacing=0.8)
        if metric == 'time':
            fig.text(0.55, 0.015, 'Separate batch runs; descriptive timing comparison.',
                     ha='center', va='bottom', fontsize=8)
        fig.canvas.draw()
        stem = output_dir / f'starting_cover_{column}_{scale}'
        metadata = {
            'Title': f'{ylabel} versus n ({scale} scale)',
            'Subject': (f'{cases} matched functions per n; '
                        + ('reported mean synthesis time in ms/function, no smoothing. '
                           if metric == 'time' else 'sum of costs, no smoothing. ')
                        +
                        'Sources and exact case intervals: starting_cover_summary.csv. '
                        'Timing is the reported batch mean, not a controlled paired comparison.'),
            'Creator': 'experiments.plot_results (Matplotlib)',
        }
        for extension in ('pdf', 'png'):
            output = stem.with_suffix('.' + extension)
            fig.savefig(output, dpi=300, metadata=metadata)
            print(output)
        plt.close(fig)


def draw_time_broken(summary, routes, break_limits, output_dir):
    """Use brokenaxes with continuous point-to-point links across the omitted band."""
    from brokenaxes import brokenaxes

    summary = summary[summary['route'].isin(routes)]
    low, high = break_limits
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
        raise ValueError('Require finite time-break limits with 0 < LOW < HIGH.')
    values = summary['avg_wall_ms']
    if ((values > low) & (values < high)).any():
        raise ValueError('The requested time break would hide observed data points.')
    if not ((values <= low).any() and (values >= high).any()):
        raise ValueError('A time break requires observations on both sides.')
    upper = math.ceil(float(values.max()) / 1000) * 1000 + 1000
    sizes = sorted(int(n) for n in summary['n'].unique())
    cases = int(summary['cases'].iloc[0])
    with plt.rc_context(PLOT_STYLE):
        fig = plt.figure(figsize=(7.2, 4.9))
        fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.23)
        # brokenaxes sizes the retained bands proportionally and draws the left slashes.
        bax = brokenaxes(ylims=((0, low), (high, upper)),
                         xlims=((min(sizes) - 0.12, max(sizes) + 0.12),),
                         hspace=0.035, d=0.008, tilt=30, despine=True, fig=fig)
        top, bottom = bax.axs
        for route in routes:
            data = summary[summary['route'] == route].sort_values('n')
            family, stage = ROUTE_STYLES[route]
            linestyle, marker = STAGE_STYLES[stage]
            native_x, native_y, bridges = [], [], []
            previous = None
            for row in data.itertuples():
                point = (row.n, row.avg_wall_ms)
                if previous is not None and (previous[1] <= low) != (point[1] <= low):
                    # Replace only the cross-band segment, not its observed endpoints.
                    native_x.append(math.nan)
                    native_y.append(math.nan)
                    bridges.append((previous, point))
                native_x.append(point[0])
                native_y.append(point[1])
                previous = point
            bax.plot(native_x, native_y, label=ROUTE_LABELS[route],
                     color=FAMILY_COLORS[family], linestyle=linestyle,
                     marker=marker, markersize=5.2, markerfacecolor='white',
                     markeredgewidth=1.1, linewidth=1.6,
                     zorder=3 if stage in ('shift', 'proxy') else 2)
            for start, stop in bridges:
                start_ax = bottom if start[1] <= low else top
                stop_ax = bottom if stop[1] <= low else top
                # A display-space guide connects the actual markers after axis compression.
                # It does not represent interpolated measurements inside the omitted range.
                bridge = ConnectionPatch(
                    xyA=start, xyB=stop, coordsA=start_ax.transData, coordsB=stop_ax.transData,
                    arrowstyle='-', connectionstyle='arc3,rad=0',
                    shrinkA=2.6, shrinkB=2.6, color=FAMILY_COLORS[family],
                    linewidth=1.6, linestyle=linestyle, clip_on=False)
                bridge.set_gid(f'{route}-cross-break-connection')
                fig.add_artist(bridge)
        for ax in (top, bottom):
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.grid(axis='y', color='#D8D8D8', linewidth=0.6)
            ax.set_axisbelow(True)
        top.set_ylim(high, upper)
        bottom.set_ylim(0, low)
        top.yaxis.set_major_locator(MaxNLocator(nbins=2))
        bottom.yaxis.set_major_locator(MaxNLocator(nbins=6))
        top.spines['bottom'].set_visible(False)
        top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        bottom.set_xticks(sizes)
        bottom.set_xlabel('Number of input variables, $n$')
        top.set_title(f'{cases:,} matched functions per input size', pad=12)
        fig.text(0.025, 0.56, METRICS['time'][1], rotation=90,
                 ha='center', va='center', fontsize=11)
        for row in summary[values >= high].itertuples():
            family, _ = ROUTE_STYLES[row.route]
            top.annotate(f'{ROUTE_LABELS[row.route]}: {row.avg_wall_ms:,.0f} ms',
                         (row.n, row.avg_wall_ms), xytext=(-10, 8),
                         textcoords='offset points', ha='right', fontsize=9,
                         color=FAMILY_COLORS[family])
        fig.legend(*bottom.get_legend_handles_labels(), loc='lower center',
                   bbox_to_anchor=(0.55, 0.035), ncol=4 if len(routes) == 8 else min(6, len(routes)),
                   frameon=False, fontsize=9.5, handlelength=LEGEND_HANDLE_LENGTH,
                   columnspacing=1.1, labelspacing=0.8)
        fig.text(0.55, 0.015, 'Separate batch runs; descriptive timing comparison.',
                 ha='center', va='bottom', fontsize=8)
        metadata = {
            'Title': 'Mean synthesis time with a broken linear axis',
            'Subject': (f'{cases} matched functions per input size; mean ms/function. '
                        f'The open interval ({low:g}, {high:g}) ms is omitted; no observed '
                        'points are hidden. Both bands use the same linear scale. '
                        'Cross-band links connect observed endpoints in display coordinates. '
                        'Source: starting_cover_summary.csv; separate batch runs.'),
            'Creator': 'experiments.plot_results (Matplotlib + brokenaxes)',
        }
        for extension in ('pdf', 'png'):
            output = output_dir / f'starting_cover_avg_wall_ms_broken.{extension}'
            fig.savefig(output, dpi=300, metadata=metadata)
            print(output)
        plt.close(fig)
    notes_path = output_dir / 'starting_cover_time_broken_notes.md'
    notes_path.write_text(
        '# Time figure with a broken axis\n\n'
        'Displayed configurations: ' + ', '.join(ROUTE_LABELS[route] for route in routes) + '.\n\n'
        f'Mean synthesis time on {cases:,} matched functions per input size. '
        f'The vertical axis omits {low:,.0f}\u2013{high:,.0f} ms/function; the retained '
        f'bands span 0\u2013{low:,.0f} and {high:,.0f}\u2013{upper:,.0f} ms/function. '
        'Both bands use the same linear scale per unit of physical height. All '
        'observed points of the displayed configurations are retained. Cross-band segments connect the observed '
        'endpoints directly in display coordinates after axis compression; they are '
        'visual guides, not interpolated measurements within the omitted interval. '
        'Only the left axis carries break marks; the narrow gap has no in-figure label. '
        'Do not interpret distances across the omitted interval as continuous.\n\n'
        'Implementation: brokenaxes manages the axes and break marks; Matplotlib '
        'ConnectionPatch draws cross-band connections. No bitmap editing is used.\n\n'
        'Timings are reported means from separate batches; circuit verification is '
        'excluded. DD0 and DD label source routes A0 and A, respectively. '
        'Exact values and source reports are in starting_cover_summary.csv.\n',
        encoding='utf-8')
    print(notes_path)


def draw_cost_comparison(summary, metric, output_dir):
    """Compare optimized routes and document the omitted raw-cover baselines."""
    if metric not in ('t', 'cx'):
        raise ValueError('Cost comparison supports only t and cx.')
    cost_column, cost_label = METRICS[metric]
    reference_column = f'reference_{cost_column}'
    normalized_column = f'normalized_{cost_column}'
    source_column = f'{metric}_source'
    cost_name = metric.upper()
    reference = summary[summary['route'] == 'F'].set_index('n')[cost_column]
    if (reference <= 0).any():
        raise ValueError('Normalization requires strictly positive F totals.')
    comparison = summary[summary['route'].isin(OPTIMIZED_ROUTES)].copy()
    comparison['reference_route'] = 'F'
    comparison[reference_column] = comparison['n'].map(reference)
    comparison[normalized_column] = (
        comparison[cost_column] / comparison[reference_column])
    comparison_path = output_dir / f'starting_cover_{metric}_comparison.csv'
    comparison.to_csv(comparison_path, index=False)
    print(comparison_path)

    fusion_rows = []
    for raw_route, route in FUSION_PAIRS:
        raw = summary[summary['route'] == raw_route].set_index('n')
        optimized = summary[summary['route'] == route].set_index('n')
        for n in sorted(raw.index):
            raw_total = int(raw.loc[n, cost_column])
            optimized_total = int(optimized.loc[n, cost_column])
            if raw_total <= 0:
                raise ValueError(f'Reduction requires a positive {raw_route} total.')
            fusion_rows.append({
                'raw_route': raw_route, 'optimized_route': route, 'n': n,
                'raw_label': ROUTE_LABELS[raw_route], 'optimized_label': ROUTE_LABELS[route],
                'start': int(raw.loc[n, 'start']), 'stop': int(raw.loc[n, 'stop']),
                'cases': int(raw.loc[n, 'cases']),
                f'raw_{cost_column}': raw_total, f'optimized_{cost_column}': optimized_total,
                'reduction_percent': 100 * (1 - optimized_total / raw_total),
                'raw_source': raw.loc[n, source_column],
                'optimized_source': optimized.loc[n, source_column],
            })
    fusion = pd.DataFrame(fusion_rows)
    fusion_path = output_dir / f'starting_cover_{metric}_fusion_reductions.csv'
    fusion.to_csv(fusion_path, index=False)
    print(fusion_path)

    sizes = sorted(int(n) for n in comparison['n'].unique())
    cases = int(comparison['cases'].iloc[0])
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), sharex=True)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.81, bottom=0.25,
                            wspace=0.40)
        for ax, column, title, ylabel in zip(
            axes, (cost_column, normalized_column),
            ('Absolute cost', 'Relative to F'),
            (cost_label, f'Normalized {cost_name} cost (F = 1)'),
        ):
            for route in OPTIMIZED_ROUTES:
                data = comparison[comparison['route'] == route].sort_values('n')
                family, stage = ROUTE_STYLES[route]
                linestyle, marker = STAGE_STYLES[stage]
                ax.plot(data['n'], data[column], label=ROUTE_LABELS[route],
                        color=FAMILY_COLORS[family], linestyle=linestyle,
                        marker=marker, markersize=4.6, markerfacecolor='white',
                        markeredgewidth=1.0, linewidth=1.5,
                        zorder=3 if stage in ('shift', 'proxy') else 2)
            ax.set_xticks(sizes)
            ax.set_xlim(min(sizes) - 0.15, max(sizes) + 0.15)
            ax.set_xlabel('Number of input variables, $n$')
            ax.set_ylabel(ylabel)
            ax.set_title(title, pad=12)
            ax.grid(axis='y', color='#D8D8D8', linewidth=0.6)
            ax.set_axisbelow(True)
        axes[0].set_ylim(bottom=0)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        axes[0].yaxis.set_major_formatter(formatter)
        axes[1].margins(y=0.09)
        fig.suptitle(f'{cases:,} matched functions per input size', y=0.97,
                     fontsize=11)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
                   bbox_to_anchor=(0.52, 0.045), ncol=8, frameon=False,
                   fontsize=9, handlelength=LEGEND_HANDLE_LENGTH, columnspacing=0.65,
                   handletextpad=0.4)
        fig.canvas.draw()
        metadata = {
            'Title': f'Optimized synthesis routes: absolute and normalized {cost_name} cost',
            'Subject': (f'{cases} matched functions per input size. Left: total {cost_name} cost. '
                        f'Right: ratio of total {cost_name} costs to F at the same n; lower is better. '
                        f'Raw baselines omitted; see starting_cover_{metric}_fusion_reductions.csv. '
                        f'Data and sources: starting_cover_{metric}_comparison.csv.'),
            'Creator': 'experiments.plot_results (Matplotlib)',
        }
        for extension in ('pdf', 'png'):
            output = output_dir / f'starting_cover_{cost_column}_comparison.{extension}'
            fig.savefig(output, dpi=300, metadata=metadata)
            print(output)
        plt.close(fig)

    ranges = []
    for raw_route, route in FUSION_PAIRS:
        reductions = fusion.loc[fusion['raw_route'] == raw_route, 'reduction_percent']
        ranges.append(f'- {ROUTE_LABELS[raw_route]} -> {ROUTE_LABELS[route]}: {reductions.min():.2f}% to '
                      f'{reductions.max():.2f}% reduction in total {cost_name} cost.')
    notes = (
        f'# {cost_name}-cost figure notes\n\n'
        f'Cohort: {cases} matched functions at each n in {sizes}; '
        f'case interval [{int(comparison["start"].iloc[0])}, '
        f'{int(comparison["stop"].iloc[0])}).\n\n'
        'Suggested caption:\n\n'
        f'{cost_name}-cost comparison of optimized synthesis configurations on {cases:,} '
        f'matched functions per input size. Left: total {cost_name} cost. Right: total {cost_name} cost '
        'normalized to F at the same input size; F = 1 and lower values are better. '
        'Colors indicate the starting-cover family, while line styles and markers '
        'indicate the optimization stage. Raw-cover baselines MT0, PK0, F0, and DD0 '
        'are omitted from these panels.\n\n'
        'DD0 and DD label source routes A0 and A, respectively.\n\n'
        'Normalization is the ratio of batch totals, not the mean of per-function '
        'ratios. The right panel uses a linear axis with a nonzero lower limit. '
        'No smoothing or offsets are applied.\n\n'
        'Improvements from raw emission to fusion-based synthesis across the displayed input sizes:\n\n'
        + '\n'.join(ranges)
        + '\n\nThese ranges describe the reductions at individual input sizes, '
        'not confidence intervals. Exact values and sources are in '
        f'starting_cover_{metric}_fusion_reductions.csv. Raw and optimized totals '
        'are retained in this CSV and in starting_cover_summary.csv. '
        'Raw baselines disable circuit segment reordering; optimized routes enable it. '
        'CX reductions therefore include both fusion and circuit segment reordering.\n'
    )
    notes_path = output_dir / f'starting_cover_{metric}_notes.md'
    notes_path.write_text(notes, encoding='utf-8')
    print(notes_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--routes', nargs='+', choices=ROUTES,
                        help='Routes to display in overview plots; time defaults to optimized routes. '
                             'Raw baselines can be selected explicitly for separate comparisons.')
    parser.add_argument('--sizes', nargs='+', type=int, default=list(range(5, 12)))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=1000)
    parser.add_argument('--metric', choices=METRICS, default='t')
    parser.add_argument('--view', choices=('auto', 'overview', 'comparison', 't-comparison'),
                        default='auto',
                        help='Default: cost comparison panels for t/cx, overview for time. '
                             't-comparison is an alias for the earlier T-only command.')
    parser.add_argument('--scale', choices=('linear', 'log', 'both'), default='linear',
                        help='Axis scale for the overview; cost comparison panels are linear.')
    parser.add_argument('--time-break', nargs=2, type=float, metavar=('LOW', 'HIGH'),
                        help='Omit an empty time interval in ms/function (linear time plot only).')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figure/results')
    args = parser.parse_args()
    source_routes = args.routes if args.routes is not None else list(ROUTES)
    plot_routes = (OPTIMIZED_ROUTES if args.metric == 'time' and args.routes is None
                   else source_routes)
    if len(set(source_routes)) != len(source_routes) or len(set(args.sizes)) != len(args.sizes):
        parser.error('Routes and input sizes must not contain duplicates.')
    if args.view == 'auto':
        args.view = 'overview' if args.metric == 'time' else 'comparison'
    if args.view == 't-comparison':
        if args.metric != 't':
            parser.error('t-comparison requires --metric t; use comparison for CX.')
        args.view = 'comparison'
    if args.view == 'comparison' and (args.metric == 'time' or set(source_routes) != set(ROUTES)):
        parser.error('comparison requires --metric t or cx and all twelve source routes.')
    if args.time_break and (args.metric != 'time' or args.scale != 'linear'):
        parser.error('--time-break requires --metric time and --scale linear.')
    summary = collect_summary(source_routes, sorted(args.sizes), args.start, args.stop)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / 'starting_cover_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(summary_path)
    if args.view == 'comparison':
        draw_cost_comparison(summary, args.metric, args.output_dir)
        return
    if args.time_break:
        draw_time_broken(summary, plot_routes, args.time_break, args.output_dir)
        return
    scales = ('linear', 'log') if args.scale == 'both' else (args.scale,)
    for scale in scales:
        draw_figure(summary, plot_routes, args.metric, scale, args.output_dir)


if __name__ == '__main__':
    main()

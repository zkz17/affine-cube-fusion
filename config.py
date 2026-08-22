config = {
    'case_pool': [
        (7, 77393837041316106211024701602844485170, 0),
        (4, 278, 1),
        (4, 1961, 1),
        (4, 1993, 1),
        (4, 2042, 1),
        (4, 2044, 1),
        (5, 1278850598, 1),
        (5, 3563652557, 1),
        (6, 2710297618828298892, 1),
        (6, 6019783381648546084, 1),
        (6, 13542117019548624888, 1),
        (7, 32249854013066286351719617629275658756, 1),
    ],
    'figure_path_prefix': 'figure/fig_',
    'figure_debug_prefix': 'figure/debug/fig_debug_',
    'debug_figure_counter': 0,
    # Cost functions for mcx gates (obtained from qiskit)
    't_cost':  lambda x: 0 if x <= 1 else 8 * (x - 1) - 1,
    'cx_cost': lambda x: 0 if x <= 0 else (1 if x == 1 else 6 * (x - 1)),
}
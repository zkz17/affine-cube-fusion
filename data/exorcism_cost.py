import sys
import time
from caterpillar_bindings import synthesize_stg_from_tt_tcost, synthesize_stg_from_tt_cxcost

numvars = int(sys.argv[1]) if len(sys.argv) >= 2 else 3
start = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
stop = int(sys.argv[3]) if len(sys.argv) >= 4 else 10000

test_cases = []
with open(f'test_cases/test_cases_{numvars}', 'r') as f:
    test_cases = f.readlines()

time_sum_t , time_avg_t  = 0, 0.0
time_sum_cx, time_avg_cx = 0, 0.0
with open(f'exorcism_cost/exorcism_t_cost_{numvars}', 'w') as f:
    if numvars <= 4:
        for i in range(1 << (1 << numvars)):
            tt = int(test_cases[i].rstrip('\n'))

            beg = time.perf_counter_ns()
            t_cost = synthesize_stg_from_tt_tcost(numvars, 0, [tt])
            end = time.perf_counter_ns()

            time_sum_t += end - beg
            f.write(f'{tt}:{t_cost}\n')
        time_avg_t = time_sum_t / (1 << (1 << numvars))
    else:
        mask = (1 << 64) - 1
        if stop > 10000: stop = 10000
        for i in range(start, stop):
            tt = int(test_cases[i].rstrip('\n'))
            tt_list, tt_tmp = [], tt
            while tt_tmp:
                tt_list.append(tt_tmp & mask)
                tt_tmp >>= 64

            beg = time.perf_counter_ns()
            t_cost = synthesize_stg_from_tt_tcost(numvars, 0, tt_list)
            end = time.perf_counter_ns()
            
            time_sum_t += end - beg
            f.write(f'{tt}:{t_cost}\n')
        time_avg_t = time_sum_t / (stop - start)

with open(f'exorcism_cost/exorcism_cx_cost_{numvars}', 'w') as f:
    if numvars <= 4:
        for i in range(1 << (1 << numvars)):
            tt = int(test_cases[i].rstrip('\n'))
            
            beg = time.perf_counter_ns()
            cx_cost = synthesize_stg_from_tt_cxcost(numvars, 0, [tt])
            end = time.perf_counter_ns()

            time_sum_cx += end - beg
            f.write(f'{tt}:{cx_cost}\n')
        time_avg_cx = time_sum_cx / (1 << (1 << numvars))
    else:
        mask = (1 << 64) - 1
        if stop > 10000: stop = 10000
        for i in range(start, stop):
            tt = int(test_cases[i].rstrip('\n'))
            tt_list, tt_tmp = [], tt
            while tt_tmp:
                tt_list.append(tt_tmp & mask)
                tt_tmp >>= 64
                
            beg = time.perf_counter_ns()
            cx_cost = synthesize_stg_from_tt_cxcost(numvars, 0, tt_list)
            end = time.perf_counter_ns()

            time_sum_cx += end - beg
            f.write(f'{tt}:{cx_cost}\n')
        time_avg_cx = time_sum_cx / (stop - start)

with open(f'exorcism_cost/exorcism_time_{numvars}', 'w') as f:
    f.write(f't_cost \n  total runtime  : {time_sum_t /1000000} ms\n  average runtime: {time_avg_t /1000000} ms\n')
    f.write(f'cx_cost\n  total runtime  : {time_sum_cx/1000000} ms\n  average runtime: {time_avg_cx/1000000} ms\n')
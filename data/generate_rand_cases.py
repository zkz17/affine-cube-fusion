import sys
import random

numvars = int(sys.argv[1]) if len(sys.argv) >= 2 else 3

with open(f'test_cases_{numvars}', 'w') as f:
    if numvars <= 4:
        for i in range(1 << (1 << numvars)):
            f.write(f'{i}\n')
    else:
        interval = (1 << (1 << numvars)) // 10
        for group in range(10):
            start = 0
            for i in range(10):
                end = start + interval
                for j in range(100):
                    tt = random.randint(start, end)
                    f.write(f'{tt}\n')
                start = end
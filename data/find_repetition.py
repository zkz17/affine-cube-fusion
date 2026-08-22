import sys

path = str(sys.argv[1]) if len(sys.argv) >= 2 else ''
with open(path, 'r') as f:
    case_map = {}
    for line in f.readlines():
        tt = int(line.rstrip('\n'))
        if tt in case_map:
            case_map[tt] += 1
            print(f'{tt} hit!')
        else: case_map[tt] = 0
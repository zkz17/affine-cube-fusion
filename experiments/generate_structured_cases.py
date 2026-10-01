"""Reproduce the DES, AES, Ascon and SPEEDY S-box component truth-table datasets.

Only the Python standard library is required. The checked-in source tables make
generation offline and reproducible; no network access or PDF parser is needed.
"""
import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data/test_cases/structured'
SOURCE_DIR = DATA_DIR / 'sources'


@dataclass(frozen=True)
class Dataset:
    n: int
    count: int
    title: str

    def filename(self, name):
        return f'test_cases_{name}_{self.n}'


DATASETS = {
    'des': Dataset(6, 32, 'DES S-boxes'),
    'aes': Dataset(8, 8, 'AES forward S-box'),
    'ascon': Dataset(5, 5, 'Ascon S-box'),
    'speedy': Dataset(6, 6, 'SPEEDY forward S-box'),
}
SOURCE_HASHES = {
    'ascon_sbox.txt': '6b4b02b193b2021560ec4d884917df7f13d085c8ba604ced23b44e482c88d4bf',
    'speedy_sbox.txt': '9b26ad158bff334854cd2dd14423a61e8c3ef33bda2349c951c6bb486c913609',
    'aes_sbox.txt': 'b0d7ef56e93f14383d0b5f4cf4ac2ff14ca000c0e39869e08a0e0017b9077985',
    'des_sboxes.txt': '6c45dd85d007640cd0839de097a6df3d14027e0a5d76e84afd67d2e74d6f69d4',
}


def source_text(name):
    text = (SOURCE_DIR / name).read_text(encoding='ascii')
    if name in SOURCE_HASHES:
        # Universal-newline reading keeps this check portable across Git CRLF settings.
        digest = hashlib.sha256(text.encode('ascii')).hexdigest()
        if digest != SOURCE_HASHES[name]:
            raise ValueError(f'Pinned source checksum mismatch: {name}')
    return text


def table_rows(name, base):
    return [[int(v, base) for v in line.split()]
            for line in source_text(name).splitlines()
            if line.strip() and not line.startswith('#')]


def component_tables(values, width):
    """Output bit j defines a scalar Boolean function; retain every bit."""
    return [sum(((value >> j) & 1) << a for a, value in enumerate(values))
            for j in range(width)]


def gf_mul(a, b):
    product = 0
    for _ in range(8):
        if b & 1:
            product ^= a
        a = ((a << 1) ^ (0x11b if a & 0x80 else 0)) & 255
        b >>= 1
    return product


def aes_sbox_algebraic(a):
    # Inverse a^254 in GF(256), with inverse(0)=0, then the AES affine map.
    inverse, power, exponent = 1, a, 254
    while exponent:
        if exponent & 1:
            inverse = gf_mul(inverse, power)
        power = gf_mul(power, power)
        exponent >>= 1
    result = inverse ^ 0x63
    for k in (1, 2, 3, 4):
        result ^= ((inverse << k) | (inverse >> (8 - k))) & 255
    return result


def generate_aes():
    rows = table_rows('aes_sbox.txt', 16)
    if len(rows) != 16 or any(len(row) != 16 for row in rows):
        raise ValueError('Invalid AES table shape')
    values = sum(rows, [])
    if values != [aes_sbox_algebraic(a) for a in range(256)]:
        raise ValueError('AES table does not match the algebraic S-box definition')
    return component_tables(values, 8), [f'Sbox[{j}]' for j in range(8)], [f'x{j}' for j in range(8)]


def generate_des():
    rows = table_rows('des_sboxes.txt', 10)
    if len(rows) != 32 or any(sorted(row) != list(range(16)) for row in rows):
        raise ValueError('Invalid DES S-box rows')
    tables, outputs = [], []
    for box in range(8):
        values = [rows[4 * box + (((a >> 5) << 1) | (a & 1))][(a >> 1) & 15]
                  for a in range(64)]
        if box == 0 and values[0b011011] != 5:
            raise ValueError('DES S1 does not match the FIPS example')
        tables.extend(component_tables(values, 4))
        outputs.extend(f'S{box + 1}[{j}]' for j in range(4))
    return tables, outputs, [f'x{j}' for j in range(6)]


def ascon_sbox_algebraic(a):
    """NIST SP 800-232, Eq. (7); specification indices are MSB first."""
    x0, x1, x2, x3, x4 = ((a >> (4 - j)) & 1 for j in range(5))
    y = (
        (x4 & x1) ^ x3 ^ (x2 & x1) ^ x2 ^ (x1 & x0) ^ x1 ^ x0,
        x4 ^ (x3 & x2) ^ (x3 & x1) ^ x3 ^ (x2 & x1) ^ x2 ^ x1 ^ x0,
        (x4 & x3) ^ x4 ^ x2 ^ x1 ^ 1,
        (x4 & x0) ^ x4 ^ (x3 & x0) ^ x3 ^ x2 ^ x1 ^ x0,
        (x4 & x1) ^ x4 ^ x3 ^ (x1 & x0) ^ x1,
    )
    return sum(bit << (4 - j) for j, bit in enumerate(y))


def speedy_sbox_dnf(a):
    """SPEEDY specification, Sec. 3.3 DNF; specification indices are MSB first."""
    x0, x1, x2, x3, x4, x5 = ((a >> (5 - j)) & 1 for j in range(6))
    # XOR with 1 complements a single bit; DNF terms combine with OR, not XOR.
    y = (
        (x3 & (x5 ^ 1)) | (x3 & x4 & x2) | ((x3 ^ 1) & x1 & x0) | (x5 & x4 & x1),
        (x5 & x3 & (x2 ^ 1)) | ((x5 ^ 1) & x3 & (x4 ^ 1))
        | (x5 & x2 & x0) | ((x3 ^ 1) & (x0 ^ 1) & x1),
        ((x3 ^ 1) & x0 & x4) | (x3 & x0 & x1) | ((x3 ^ 1) & (x4 ^ 1) & x2)
        | ((x0 ^ 1) & (x2 ^ 1) & (x5 ^ 1)),
        ((x0 ^ 1) & x2 & (x3 ^ 1)) | (x0 & x2 & x4) | (x0 & (x2 ^ 1) & x5)
        | ((x0 ^ 1) & x3 & x1),
        (x0 & (x3 ^ 1)) | (x0 & (x4 ^ 1) & (x2 ^ 1)) | ((x0 ^ 1) & x4 & x5)
        | ((x4 ^ 1) & (x2 ^ 1) & x1),
        (x2 & x5) | ((x2 ^ 1) & (x1 ^ 1) & x4) | (x2 & x1 & x0)
        | ((x1 ^ 1) & x0 & x3),
    )
    return sum(bit << (5 - j) for j, bit in enumerate(y))


def generate_checked_sbox(name, definition):
    n = DATASETS[name].n
    rows = table_rows(f'{name}_sbox.txt', 16)
    if len(rows) != (1 << n) // 16 or any(len(row) != 16 for row in rows):
        raise ValueError(f'Invalid {name} table shape')
    values = sum(rows, [])
    if sorted(values) != list(range(1 << n)):
        raise ValueError(f'{name} S-box is not a permutation')
    if values != [definition(a) for a in range(1 << n)]:
        raise ValueError(f'{name} table does not match the Boolean S-box definition')
    return component_tables(values, n), [f'Sbox[{j}]' for j in range(n)], [f'x{j}' for j in range(n)]


def generate(name):
    if name == 'des':
        result = generate_des()
    elif name == 'aes':
        result = generate_aes()
    elif name == 'ascon':
        result = generate_checked_sbox(name, ascon_sbox_algebraic)
    elif name == 'speedy':
        result = generate_checked_sbox(name, speedy_sbox_dnf)
    else:
        raise ValueError(f'Unknown S-box dataset: {name}')
    tables, outputs, inputs = result
    spec = DATASETS[name]
    if len(inputs) != spec.n or len(tables) != spec.count or len(outputs) != spec.count:
        raise ValueError(f'Unexpected dimensions: {name}')
    return result


def load_cases(name, directory=DATA_DIR):
    spec = DATASETS[name]
    path = Path(directory) / spec.filename(name)
    cases = [int(line) for line in path.read_text(encoding='ascii').splitlines()]
    if len(cases) != spec.count or any(not 0 <= tt < (1 << (1 << spec.n)) for tt in cases):
        raise ValueError(f'Invalid dataset: {path}')
    # Verify the full ordered dataset, not only its length or its support sizes.
    if cases != generate(name)[0]:
        raise ValueError(f'Dataset differs from pinned sources: {path}')
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check existing files without changing them')
    parser.add_argument('--overwrite', action='store_true', help='Replace differing generated files')
    parser.add_argument('--output-dir', type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    generated = {}
    mapping = ['dataset,line_1based,output']
    for name, spec in DATASETS.items():
        tables, outputs, inputs = generate(name)
        generated[spec.filename(name)] = ''.join(f'{tt}\n' for tt in tables)
        mapping.extend(f'{name},{j},{output}' for j, output in enumerate(outputs, 1))
        print(f'{name}: n={len(inputs)}, outputs={len(tables)}, distinct={len(set(tables))}, '
              f'constant={sum(tt in (0, (1 << (1 << spec.n)) - 1) for tt in tables)}')
    generated['case_index.csv'] = '\n'.join(mapping) + '\n'
    # Preflight the entire set before any writes.
    for filename, content in generated.items():
        path = args.output_dir / filename
        same = path.exists() and path.read_text(encoding='ascii') == content
        if args.check and not same:
            raise ValueError(f'Missing or stale generated file: {path}')
        if not args.check and path.exists() and not same and not args.overwrite:
            raise FileExistsError(f'Refusing to replace {path}; inspect changes before --overwrite')
    if not args.check:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in generated.items():
            (args.output_dir / filename).write_text(content, encoding='ascii', newline='\n')
    print('All source and truth-table checks passed.')


if __name__ == '__main__':
    main()

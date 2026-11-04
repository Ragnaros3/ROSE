"""Generate the published one-view-drop masks and check their array hashes."""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rose.protocol import make_one_view_drop_mask

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'masks/one_view_drop')
    args = parser.parse_args()
    with (ROOT/'masks/mask_manifest.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        n, v = map(int, row['shape'].split('x'))
        mask = make_one_view_drop_mask(n, v, float(row['eta']), int(row['seed'])).astype(np.uint8)
        digest = hashlib.sha256(mask.tobytes(order='C')).hexdigest()
        if digest != row['array_sha256']:
            raise RuntimeError('Generated mask differs from the protocol: '+row['file'])
        path = args.output_dir/Path(row['file']).name
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                if archive.files != ['mask'] or archive['mask'].dtype != mask.dtype or not np.array_equal(archive['mask'], mask):
                    raise RuntimeError('Existing mask differs; left unchanged: '+str(path))
        else:
            np.savez_compressed(path, mask=mask)
    print(f'Generated or verified {len(rows)} masks in {args.output_dir}')

if __name__ == '__main__':
    main()

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
from collections import Counter
p = Path('data/processed')
print('Checking split files in', p)
for s in ('train','val','test'):
    path = p/f'{s}.txt'
    print(s, 'exists?', path.exists())

train = set(open(p/'train.txt').read().splitlines())
val = set(open(p/'val.txt').read().splitlines())
test = set(open(p/'test.txt').read().splitlines())
print('train∩val:', len(train & val))
print('train∩test:', len(train & test))
print('val∩test:', len(val & test))

all_files = list(p.glob('*.png'))
cn = Counter(['_aug' in f.stem for f in all_files])
print('has_aug Count:', cn)
print('Total png files:', len(all_files))
print('Sample files:', [f.name for f in all_files[:10]])

# Check label parsing for first 20 files using current EmotionDataset parser
import sys, os
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
from lightspeech.code.data.loader import load_dataset
files = load_dataset('data/processed')
print('load_dataset found', len(files), 'files')
for f in files[:20]:
    name = Path(f).name
    parts = name.split('-')
    emo_id = parts[2] if len(parts) > 2 else '<missing>'
    print(name, '-> emo_id:', emo_id)

# compute unique emotion ids from base names
emo_ids = {}
for f in files:
    base = Path(f).name.split('_')[0]
    parts = base.split('-')
    if len(parts) > 2:
        emo = parts[2]
        emo_ids.setdefault(emo, 0)
        emo_ids[emo] += 1

print('Unique emotion ids and counts:', emo_ids)

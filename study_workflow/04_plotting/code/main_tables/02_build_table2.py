"""Generate Table 2 from the fixed indicator weights."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from table_utils import Tables, fixed_weights, INDICATORS
weights, sources = fixed_weights()
book = Tables('Table2.xlsx', sources)
rows = []
for indicator in INDICATORS:
    label = 'Albedo proxy (A)' if indicator == 'Albedo' else indicator
    rows.append([label, weights['with_NPV'][indicator], weights['without_NPV'].get(indicator, '—')])
book.add('Table2', rows, {1: '0.000', 2: '0.000'}, 'Table2', widths=[38, 24, 24])
book.save()

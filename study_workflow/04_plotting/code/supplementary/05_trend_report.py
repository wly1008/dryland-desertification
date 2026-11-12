"""Write a numerical summary of the OLS versus Sen-MK-BH comparison."""
from pathlib import Path
import json, sys
root=next(p for p in Path(__file__).resolve().parents if (p/'07_environment').is_dir())
sys.path.insert(0,str(root/'07_environment'))
from project_paths import PROJECT_ROOT
folder=PROJECT_ROOT/'03_analysis/07_sensitivity/data/presentation/trend_robustness'
statistics=json.loads((folder/'statistics.json').read_text(encoding='utf-8'))
body='# Trend-method sensitivity results\n\n'
body+='A: OLS with nominal P < 0.05. D: Theil-Sen, Mann-Kendall, and BH q < 0.05.\n\n'
body+='```json\n'+json.dumps(statistics,indent=2)+'\n```\n'
(folder/'trend_sensitivity.md').write_text(body,encoding='utf-8')

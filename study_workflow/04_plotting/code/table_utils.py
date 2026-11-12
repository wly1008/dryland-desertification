"""Provide shared XLSX table formatting with openpyxl."""
from pathlib import Path
import hashlib, json, math, textwrap
from numbers import Real
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
_CODE = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(_CODE/'07_environment'))
from project_paths import PROJECT_ROOT, parameter_file
ROOT = PROJECT_ROOT
OUTPUT = ROOT / '05_outputs/tables'
META = parameter_file('04_plotting/intermediate/publication_tables/table_layouts.json')
FONT = 'Arial'
INDICATORS = ['Albedo', 'SMI', 'TGSI', 'Wet', 'LST', 'PV', 'NPV']
SCENARIOS = ['entropy_G15', 'entropy_G10', 'entropy_G20', 'equal']
LABELS = {'entropy_G15': 'G = 15 (main)', 'entropy_G10': 'G = 10', 'entropy_G20': 'G = 20', 'equal': 'Equal'}
PRODUCTS = ['FAO–WOCAT', 'JRC', 'Trends.Earth']
METHODS = ['A_OLS', 'D_Sen_MK_BH']
METHOD_LABELS = {'A_OLS': 'OLS (main)', 'D_Sen_MK_BH': 'Sen–MK–BH'}
SIGNED4 = '+0.0000;-0.0000;0.0000'
SIGNED3 = '+0.000;-0.000;0.000'
SIGNED2 = '+0.00;-0.00;0.00'

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def fixed_weights():
    """Read the exact study-period W columns used to construct the annual DSI."""
    base = ROOT / '03_analysis/01_dsi/data/01indices/02indicator_groups'
    values = {}
    sources = []
    for variant, suffix in [('with_NPV', '_NPV'), ('without_NPV', '')]:
        p = base / ('Albedo_SMI_TGSI_Wet_LST_PV' + suffix) / '06Gk_Er_W/Gk_Er_W.xlsx'
        wb = load_workbook(p, read_only=True, data_only=True)
        rows = iter(wb['W'].values)
        header = next(rows)
        column = header.index('study_period')
        weights = {r[0]: float(r[column]) for r in rows if r[0]}
        assert math.isclose(sum(weights.values()), 1, abs_tol=1e-12)
        values[variant] = weights
        sources.append(p)
        wb.close()
    return (values, sources)

class Tables:

    def __init__(self, filename, sources):
        self.filename = filename
        self.sources = [Path(p) for p in sources]
        self.wb = Workbook()
        self.wb.remove(self.wb.active)
        self.metadata = read_json(META)
        self.expected = []

    def add(self, ident, rows, formats=None, sheet_name=None, widths=None, bold_delta=False):
        spec = self.metadata[ident]
        headers = spec['headers']
        n = len(headers)
        assert rows and all((len(r) == n for r in rows)), (ident, 'wrong number of columns')
        ws = self.wb.create_sheet(sheet_name or ident)
        ws.sheet_view.showGridLines = False
        if widths is None:
            widths = [26 if j < 2 else 24 for j in range(n)]
        last = get_column_letter(n)
        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=n)
        ws.cell(2, 1, spec['caption']).font = Font(name=FONT, size=12, bold=True)
        ws.cell(2, 1).alignment = Alignment(wrap_text=True, vertical='center')
        ws.row_dimensions[2].height = 43
        ws.row_dimensions[1].height = 10
        ws.row_dimensions[3].height = 10
        ws.append([])
        for j, h in enumerate(headers, 1):
            ws.cell(5, j, h)
        ws.row_dimensions[5].height = 64
        top = Side(style='thin', color='333333')
        for j, width in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(j)].width = width
        for cell in ws[5]:
            cell.font = Font(name=FONT, size=10, bold=True)
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = Border(top=top, bottom=top)
        for i, row in enumerate(rows, 6):
            line_count = 1
            for j, value in enumerate(row, 1):
                if isinstance(value, Real):
                    assert math.isfinite(value), (ident, i, j, value)
                cell = ws.cell(i, j, value)
                cell.font = Font(name=FONT, size=10)
                cell.alignment = Alignment(horizontal='right' if isinstance(value, Real) else 'left', vertical='center', wrap_text=True)
                if formats and j - 1 in formats:
                    cell.number_format = formats[j - 1]
                line_count = max(line_count, sum((max(1, math.ceil(len(s) / (widths[j - 1] * 0.85))) for s in str(value).split('\n'))))
            ws.row_dimensions[i].height = max(26, line_count * 15 + 9)
            if all((v in ('', None) for v in row[1:])):
                ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=n)
                ws.cell(i, 1).font = Font(name=FONT, size=10, bold=True)
            if bold_delta and isinstance(row[-2], Real) and isinstance(row[-1], Real) and (row[-2] * row[-1] > 0):
                ws.cell(i, n - 2).font = Font(name=FONT, size=10, bold=True)
        end = 5 + len(rows)
        for cell in ws[end]:
            cell.border = Border(bottom=top)
        for k, note in enumerate(spec.get('notes', []), end + 2):
            ws.merge_cells(start_row=k, start_column=1, end_row=k, end_column=n)
            cell = ws.cell(k, 1, note)
            cell.font = Font(name=FONT, size=9)
            cell.alignment = Alignment(wrap_text=True, vertical='center')
            ws.row_dimensions[k].height = max(28, math.ceil(len(note) / (sum(widths) * 0.8)) * 14 + 8)
        ws.freeze_panes = 'C6' if n > 5 else 'B6'
        ws.print_options.horizontalCentered = True
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.orientation = 'landscape'
        ws.page_setup.paperSize = ws.PAPERSIZE_A4
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.print_title_rows = '1:5'
        ws.print_area = f"A1:{last}{end + len(spec.get('notes', [])) + 2}"
        self.expected.append({'id': ident, 'sheet': ws.title, 'range': f'A6:{last}{end}', 'values': rows, 'formats': formats or {}})

    def save(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path = OUTPUT / self.filename
        temporary = path.with_suffix('.part.xlsx')
        self.wb.save(temporary)
        temporary.replace(path)
        reread = load_workbook(path, read_only=True, data_only=True)
        for table in self.expected:
            actual = [[c.value for c in row] for row in reread[table['sheet']][table['range']]]
            expected = [[None if v == '' else v for v in row] for row in table['values']]
            for ar, er in zip(actual, expected):
                for a, e in zip(ar, er):
                    if isinstance(e, Real):
                        assert isinstance(a, Real) and math.isclose(a, e, rel_tol=1e-14, abs_tol=1e-14), (table['id'], a, e)
                    else:
                        assert a == e, (table['id'], a, e)
        reread.close()
        manifest = {'workbook': path.relative_to(ROOT).as_posix(), 'sources': [{'path': (p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else '__code__/' + p.relative_to(_CODE).as_posix()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in [META, *self.sources]], 'tables': self.expected, 'numeric_cells': 'full precision; manuscript display precision applied using XLSX number formats', 'source': 'analysis results and documented data-source metadata; manuscript rows are not inputs'}
        log = ROOT / '06_verification/python_tables' / Path(self.filename).stem
        log.parent.mkdir(parents=True, exist_ok=True)
        log.with_suffix('.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Saved {path}: {len(self.expected)} tables', flush=True)
        return path

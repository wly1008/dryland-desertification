"""Check, rebuild, or rebase source VRT references."""
from pathlib import Path
import argparse, json, sys, xml.etree.ElementTree as ET
ROOT = next((p for p in Path(__file__).resolve().parents if (p / '07_environment').is_dir()))
sys.path.insert(0, str(ROOT/'07_environment'))
from project_paths import PROJECT_ROOT
ROOT = PROJECT_ROOT
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--rebuild', action='store_true')
p.add_argument('--rebase', action='append', default=[], metavar='OLD=NEW')
a = p.parse_args()
pairs = [tuple(x.split('=', 1)) for x in a.rebase]

def rebase(value):
    s = str(value).replace(chr(92), '/')
    for old, new in pairs:
        old = old.replace(chr(92), '/').rstrip('/')
        if s.lower() == old.lower() or s.lower().startswith(old.lower() + '/'):
            return Path(new) / s[len(old):].lstrip('/')
    return Path(value)
rows = json.loads((ROOT / '06_verification/external_vrt_manifest.json').read_text(encoding='utf-8'))
if a.rebuild:
    from osgeo import gdal
    gdal.UseExceptions()
    for row in rows:
        source = rebase(row['source'])
        dest = ROOT / row['vrt']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            raise FileNotFoundError(source)
        if '/lst/' in row['vrt']:
            t = ET.parse(source)
            for n in t.iter('SourceFilename'):
                q = (source.parent / n.text).resolve() if n.attrib.get('relativeToVRT') == '1' else Path(n.text)
                q = rebase(str(q).replace(chr(92) + '03orig' + chr(92), chr(92) + '00orig' + chr(92)))
                n.text = str(q)
                n.set('relativeToVRT', '0')
            t.write(dest, encoding='utf-8', xml_declaration=True)
        else:
            ds = gdal.Translate(str(dest), str(source), format='VRT')
            ds = None
if pairs:
    for row in rows:
        dest = ROOT / row['vrt']
        t = ET.parse(dest)
        changed = False
        for n in t.iter('SourceFilename'):
            q = (dest.parent / n.text).resolve() if n.attrib.get('relativeToVRT') == '1' else Path(n.text)
            new = rebase(q)
            if new != q:
                n.text = str(new)
                n.set('relativeToVRT', '0')
                changed = True
        if changed:
            t.write(dest, encoding='utf-8', xml_declaration=True)
missing = []
seen = set()

def check(dest):
    dest = dest.resolve()
    if dest in seen:
        return
    seen.add(dest)
    for n in ET.parse(dest).iter('SourceFilename'):
        q = (dest.parent / n.text).resolve() if n.attrib.get('relativeToVRT') == '1' else Path(n.text)
        if not q.is_file():
            missing.append({'vrt': str(dest), 'source': str(q)})
        elif q.suffix.lower() == '.vrt':
            check(q)
for row in rows:
    check(ROOT / row['vrt'])
report = {'archive_vrts': len(rows), 'including_nested_vrts': len(seen), 'missing': missing}
(ROOT / '06_verification/external_reference_current_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'archive_vrts': len(rows), 'missing_sources': len(missing)}))
raise SystemExit(1 if missing else 0)

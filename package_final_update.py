"""Create complete and incremental delivery archives, without changing experiments.

Run after report generation/visual review. Existing output archives are protected.
All raw DeepDish input files must remain byte-identical to the supplied result ZIP.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def include(path, root):
    rel = path.relative_to(root)
    return (path.is_file() and path.suffix not in {'.zip', '.pyc', '.tmp'}
            and rel.parts[0] not in {'output', '.git', 'checks_rerun', 'fresh_cartpole', 'fresh_deepdish', 'probe_rerun'}
            and not any(part.startswith('.') for part in rel.parts if part != '.gitignore')
            and '__pycache__' not in rel.parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--input-results', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.destination.resolve()
    output.mkdir(parents=True, exist_ok=True)
    full_zip = output/'Assignment1_RL_Submission.zip'
    update_zip = output/'Assignment1_RL_Update.zip'
    pdf = output/'Assignment1_RL_Report.pdf'
    assert not any(p.exists() for p in (full_zip, update_zip, pdf)), 'Use a fresh destination.'
    for path in root.rglob('*.py'):
        if include(path, root):
            ast.parse(path.read_text())
    with zipfile.ZipFile(args.input_results) as source:
        assert source.testzip() is None
        matched = 0
        for item in source.infolist():
            if item.is_dir():
                continue
            assert source.read(item.filename) == (root/'deepdish_runs'/item.filename).read_bytes()
            matched += 1
    manifest = {}
    for path in sorted(root.rglob('*')):
        if include(path, root) and path.name != 'DELIVERY_MANIFEST.json':
            manifest[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size, sha256=digest(path))
    (root/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    files = [p for p in sorted(root.rglob('*')) if include(p, root)]
    with zipfile.ZipFile(full_zip, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, 'Assignment1_RL_Submission/'+path.relative_to(root).as_posix())
    root_updates = {'Assignment1_RL_Report.pdf', 'README.md', 'START_HERE.ipynb',
                    'audit_deepdish.py', 'build_submission_pdf.py', 'deepdish_pong.py',
                    'verify_submission.py', 'finalize_submission.py', 'submission_identity.json',
                    'package_notes.json', 'PACKAGING_QA.json', 'DELIVERY_MANIFEST.json',
                    'DEEPDISH_UPDATE_NOTES.md', 'package_final_update.py'}
    count = 0
    with zipfile.ZipFile(update_zip, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            rel = path.relative_to(root).as_posix()
            if rel in root_updates or rel.startswith('deepdish_runs/') or rel.startswith('review/deepdish_audit/'):
                archive.write(path, 'Assignment1_RL_Update/'+rel)
                count += 1
    shutil.copy2(root/'Assignment1_RL_Report.pdf', pdf)
    for path in (full_zip, update_zip):
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            assert all(not name.startswith('/') and '..' not in Path(name).parts for name in archive.namelist())
    result = dict(raw_deepdish_files_preserved=matched, complete_file_count=len(files),
                  update_file_count=count, training_performed=False,
                  files={p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in (pdf, update_zip, full_zip)})
    (output/'delivery_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Run fast checks and save numerical evidence. No game training."""
import argparse
import csv
import importlib.metadata
import io
import json
import platform
import unittest
from pathlib import Path

from project_code import load_project
import test_rl

def main(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    buffer=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_rl)
    result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
    print(buffer.getvalue())
    (output/'test_log.txt').write_text(buffer.getvalue())
    rows=test_rl.analytical_rows()
    with (output/'analytical_gradients.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    versions={'python':platform.python_version()}
    for name in ['numpy','torch','gymnasium','ale-py','pandas','matplotlib','reportlab']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='not installed'
    summary=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                 passed=result.wasSuccessful(),analytical_cases=len(rows),
                 max_gradient_error=max(r['max_absolute_error'] for r in rows),
                 source_sha256=load_project()['source_sha256'],
                 versions=versions,training_performed=False,
                 scope='Unit and independent analytical checks; not proof of game-learning success')
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'requirements-validation.txt').write_text('\n'.join(
        f'{k}=={v}' for k,v in versions.items() if k!='python' and v!='not installed')+'\n')
    print('Evidence saved in:',output)
    if not result.wasSuccessful(): raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='validation_results')
    main(parser.parse_args().output)


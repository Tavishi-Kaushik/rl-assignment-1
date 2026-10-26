"""Add actual repository identity and package existing work; never publish/train."""
import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def main(url):
    match=re.fullmatch(r'https://github\.com/([^/\s]+)/([^/\s]+)/?',url.strip())
    if not match or any('YOUR_' in part for part in match.groups()):
        raise ValueError('Enter your actual GitHub repository URL, not the example.')
    root=Path(__file__).resolve().parent
    url=url.rstrip('/')
    subprocess.run([sys.executable,str(root/'build_submission_pdf.py'),'--repo',url],check=True)
    shutil.copy2(root/'output/pdf/Assignment1_RL_Report.pdf',root/'Assignment1_RL_Report.pdf')
    identity={'author':'Tavishi Kaushik','github_username':match.group(1),
              'repository_name':match.group(2),'repository_url':url,
              'repository_published_or_access_verified_by_script':False}
    (root/'submission_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    notes_path=root/'package_notes.json'
    if notes_path.exists():
        notes=json.loads(notes_path.read_text())
        notes.update(github_url_missing=False,repository_url=url,
                     required_next_action='Upload source and verify grader access before Canvas submission.')
        notes_path.write_text(json.dumps(notes,indent=2)+'\n')
    readme_path=root/'README.md'
    if readme_path.exists():
        readme=readme_path.read_text()
        marker='\n## Recorded submission identity\n'
        readme=readme.split(marker)[0]
        readme=readme.replace('The actual GitHub username, repository name and URL were not supplied, so those fields remain deliberately blank.',
                             'The repository details were supplied during finalization and are printed in the regenerated report.')
        readme_path.write_text(readme+marker+'\nGitHub username: '+match.group(1)+
                              '\n\nRepository name: '+match.group(2)+'\n\nURL: '+url+'\n')
    archive=root/'GitHub_Upload.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob('*')):
            rel=path.relative_to(root)
            if (path.is_file() and path.suffix not in {'.zip','.pyc','.tmp'}
                and '__pycache__' not in rel.parts
                and rel.parts[0] not in {'output','.git','checks_rerun','fresh_cartpole','fresh_deepdish','probe_rerun'}
                and not any(part.startswith('.') for part in rel.parts if part != '.gitignore')):
                z.write(path,rel.as_posix())
    print('Updated:',root/'Assignment1_RL_Report.pdf')
    print('Created:',archive)
    print('Review the PDF. Upload the extracted folder contents to your repository, '
          'verify grader access, then submit the PDF on Canvas. No training performed.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True)
    main(ap.parse_args().repo)

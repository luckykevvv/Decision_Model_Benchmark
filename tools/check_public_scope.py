"""Fail if the Git index contains anything outside the public benchmark."""
from pathlib import Path, PurePosixPath
import subprocess

ROOT=Path(__file__).resolve().parents[1]
TOP={'.env.example','.gitattributes','.gitignore','README.md','LICENSE','CITATION.cff','pyproject.toml'}
PUBLIC_TOOLS={'__init__.py','figure_layout_check.py','check_public_scope.py'}
def allowed(name):
    path=PurePosixPath(name)
    if name in TOP:return True
    if name.startswith('.github/workflows/') and path.suffix in ('.yml','.yaml'):return True
    if name.startswith('decision_benchmark/resources/') and path.suffix in ('.json','.gz'):return True
    if path.suffix!='.py':return False
    if path.parts[0] in ('decision_benchmark','systematic','tests'):return True
    return path.parts[0]=='tools' and len(path.parts)==2 and path.name in PUBLIC_TOOLS

def main():
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    bad=[name for name in names if name and not allowed(name)]
    if bad:raise SystemExit('Non-benchmark files tracked:\n'+'\n'.join(bad))
    print('PASS: Git index contains only the public benchmark allowlist.')

if __name__=='__main__':main()

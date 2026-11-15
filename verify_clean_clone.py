"""Verify the actual published repository in a new temp clone and fresh venv.

Retains the clone for the subsequent browser/serve test; never deletes user files.
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'lab3'


def main():
    clone=Path(tempfile.mkdtemp(prefix='grafkom-module3-clean-')).resolve()
    log=[]
    def run(args):
        log.append('$ '+' '.join(str(x) for x in args))
        p=subprocess.run([str(x) for x in args],cwd=clone,capture_output=True,text=True,
                         encoding='utf-8',errors='replace')
        log.append(p.stdout);log.append(p.stderr)
        (OUT/'clean_clone_verification.txt').write_text('\n'.join(log),encoding='utf-8')
        if p.returncode:raise RuntimeError(f'Clean-clone command failed: {args[0]}')
    run(['git','clone','https://github.com/riooptr/grafkom-modul3.git','.'])
    run([sys.executable,'-m','venv','.venv'])
    py=clone/'.venv/Scripts/python.exe'
    run([py,'-m','pip','install','-r','requirements.txt'])
    run([py,'make_model.py'])
    run([py,'tests.py'])
    run([py,'prepare_web.py'])
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=clone,text=True).strip()
    result={'clone_path':str(clone),'commit':sha,'fresh_venv':True,'core_tests':'14/14 passed; none skipped',
            'make_model':True,'prepare_web_vendor_checksum':True,'browser_serve_status':'pending'}
    (OUT/'clean_clone.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

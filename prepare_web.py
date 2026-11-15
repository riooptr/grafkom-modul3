"""Prepare a self-contained web/ folder and pin the Three.js r160 modules."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import urllib.request

ROOT=Path(__file__).resolve().parent
FILES={'three.module.js':'build/three.module.js',
       'addons/controls/OrbitControls.js':'examples/jsm/controls/OrbitControls.js',
       'addons/loaders/OBJLoader.js':'examples/jsm/loaders/OBJLoader.js',
       'LICENSE.three.txt':'LICENSE'}


def prepare(refresh=False):
    assets=ROOT/'web/assets';assets.mkdir(parents=True,exist_ok=True)
    for name in ['mymodel.obj','mymodel.mtl','lab1_brick.png','ASSET_LICENSE.md']:
        shutil.copyfile(ROOT/'data'/name,assets/name)
    vendor=ROOT/'web/vendor'
    manifest_path=vendor/'manifest.json'
    previous=(json.loads(manifest_path.read_text(encoding='utf-8')).get('sha256',{})
              if manifest_path.exists() and not refresh else {})
    for dest,source in FILES.items():
        path=vendor/dest
        if not path.exists() or refresh:
            path.parent.mkdir(parents=True,exist_ok=True)
            with urllib.request.urlopen(f'https://cdn.jsdelivr.net/npm/three@0.160.0/{source}',timeout=60) as response:
                path.write_bytes(response.read())
        if dest in previous and hashlib.sha256(path.read_bytes()).hexdigest()!=previous[dest]:
            raise RuntimeError(f'Vendor checksum changed: {dest}; inspect before refreshing')
    manifest={'three_version':'0.160.0','source':'https://www.npmjs.com/package/three/v/0.160.0',
              'sha256':{n:hashlib.sha256((vendor/n).read_bytes()).hexdigest() for n in FILES}}
    (vendor/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (ROOT/'web/.nojekyll').touch()
    print('Prepared local assets and Three.js 0.160.0; no CDN required at runtime')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--refresh-vendor',action='store_true')
    prepare(ap.parse_args().refresh_vendor)

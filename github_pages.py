"""Inspect/configure the explicitly supplied GitHub Pages repository via cached GCM.

Credentials remain in process memory and are sent only to api.github.com.
Never print, save, or put a token in command-line arguments or remote URLs.
"""
import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request

REPO='riooptr/grafkom-modul3'


def credential():
    env={**os.environ,'GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'never'}
    proc=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\nusername=riooptr\n\n',
                        text=True,capture_output=True,env=env)
    if proc.returncode:raise RuntimeError('GitHub credential unavailable; authenticate through Git Credential Manager')
    values=dict(line.split('=',1) for line in proc.stdout.splitlines() if '=' in line)
    if not values.get('password'):raise RuntimeError('No GitHub token returned by credential manager')
    return values['password']


def api(path,method='GET',body=None,token=None):
    req=urllib.request.Request('https://api.github.com/'+path,method=method,
        headers={'Accept':'application/vnd.github+json','Authorization':'Bearer '+token,
                 'X-GitHub-Api-Version':'2026-03-10','User-Agent':'Module3-Lab3-Pages',
                 'Content-Type':'application/json'},
        data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req,timeout=30) as response:
            data=response.read();return json.loads(data) if data else {}
    except urllib.error.HTTPError as error:
        try:message=json.loads(error.read()).get('message',error.reason)
        except Exception:message=str(error.reason)
        return {'http_status':error.code,'message':message}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--enable',action='store_true');ap.add_argument('--status',action='store_true')
    args=ap.parse_args();token=credential()
    user=api('user',token=token)
    repo=api('repos/'+REPO,token=token)
    print(json.dumps({'authenticated_login':user.get('login'),'repository':repo.get('full_name'),
                      'private':repo.get('private'),'permissions':repo.get('permissions'),
                      'default_branch':repo.get('default_branch'),'size':repo.get('size'),
                      'repository_error':repo.get('message')},indent=2))
    if args.enable:
        pages=api('repos/'+REPO+'/pages',token=token)
        if pages.get('http_status')==404:
            pages=api('repos/'+REPO+'/pages','POST',{'source':{'branch':'gh-pages','path':'/'}},token)
        print(json.dumps({k:pages.get(k) for k in ['html_url','status','source','http_status','message']},indent=2))
    if args.status:
        pages=api('repos/'+REPO+'/pages',token=token)
        build=api('repos/'+REPO+'/pages/builds/latest',token=token)
        print(json.dumps({'pages':{k:pages.get(k) for k in ['html_url','status','source','http_status','message']},
                          'latest_build':{k:build.get(k) for k in ['status','error','commit','duration','http_status','message']}},indent=2))


if __name__=='__main__':main()

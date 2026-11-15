"""Offline Windows launcher, A02 only, no account secrets written to disk."""
import argparse
from pathlib import Path
import secrets
import webbrowser
from stage_p_a02_portable_revision_v1 import RevisionStudy,Server

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8769);parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    study=RevisionStudy(root/'WORKSPACE',root/'STATE')
    # Bind before provisioning, so an occupied port creates no accidental account.
    server=Server(('127.0.0.1',args.port),study,root/'WEB')
    with study.connect() as db:exists=db.execute("SELECT 1 FROM users WHERE id='A02'").fetchone()
    if not exists:
        words='River Maple Meadow Cedar Willow Ocean Clover Autumn Sparrow Orchard Amber Breeze Brook Cherry Daisy Finch Hazel Laurel Lemon Lotus Olive Orchid Otter Peach Pearl Pine Poppy Robin Sage Sunset Tulip Valley Violet Wren'.split()
        password='-'.join(secrets.choice(words) for _ in range(4))+'-'+str(secrets.randbelow(90)+10)
        assert study.add_user('A02','rater',password)
        print('PRIVATE LOGIN - record now; plaintext is not saved:',flush=True)
        print('Account: A02\nPassword: '+password,flush=True)
    else:print('A02 account retained. Use the password printed on first launch.',flush=True)
    url='http://127.0.0.1:'+str(args.port)+'/'
    print(url+' - A02 revision only. Keep this window open. Ctrl+C stops the server.',flush=True)
    if not args.no_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()

"""Windows/CPU launcher. Default preview; collection only when operator explicitly starts it."""
import argparse,json,secrets,webbrowser,sys
from pathlib import Path
from stage_p_human_evaluation_v1 import Study,Server,live_release_valid
from human_app_core_v1 import file_hash
from human_http_common_v1 import now
def main():
    p=argparse.ArgumentParser();p.add_argument('--open-session',action='store_true');p.add_argument('--collect',action='store_true');p.add_argument('--no-browser',action='store_true');p.add_argument('--port',type=int,default=8768);p.add_argument('--workspace',type=Path);p.add_argument('--state-dir',type=Path);a=p.parse_args()
    base=Path(__file__).resolve().parents[1];w=(a.workspace or base/'WORKSPACE').resolve();state=(a.state_dir or base/'STATE').resolve()
    if a.open_session:
        marker=w/'COLLECTION_OPEN.json'
        if not marker.exists():
            with marker.open('x',encoding='utf8') as f:json.dump(dict(operator_started_session=True,opened_at_utc=now(),config_sha256=file_hash(w/'CONFIG.json'),protocol_sha256=file_hash(w/'TWO_RATER_PROTOCOL_AMENDMENT.md'),authority='Local operator explicitly launched START_COLLECTION after reviewing this prepared package; no participant declarations or scores are inferred.'),f,indent=2)
        a.collect=True
    if a.collect and not live_release_valid(w):raise SystemExit('Collection is not open. Review first, then use START_COLLECTION.cmd. No scores changed.')
    study=Study(w,preview=not a.collect,state_dir=state)
    words='Maple River Lantern Meadow Cedar Harbor Willow Cloud Garden Pebble Forest Sunlight Ocean Clover Silver Autumn Sparrow Summit Orchard Velvet Coral Amber Breeze Morning Birch Brook Cabin Cherry Comet Copper Daisy Fern Finch Grove Hazel Indigo Island Jade Lake Laurel Lemon Lilac Lotus Mango Olive Opal Orchid Otter Peach Pearl Pine Plum Poppy Raven Reed Robin Sage Sand Shell Spruce Stone Sunset Swan Tulip Valley Violet Water Wheat Wind Wren'.split()
    generated=set()
    for identity,role in [('A01','rater'),('A02','rater'),('COORDINATOR','coordinator')]:
        password='-'.join(secrets.choice(words) for _ in range(4))+'-'+str(secrets.randbelow(90)+10)
        while password in generated:password='-'.join(secrets.choice(words) for _ in range(4))+'-'+str(secrets.randbelow(90)+10)
        generated.add(password)
        if study.add_user(identity,role,password):print(identity+' | '+password,flush=True)
        else:print(identity+' | Existing account retained; password unchanged.',flush=True)
    print('Record new passwords privately now. They are hashed in STATE, never written as plaintext to a package or report.',flush=True)
    try:server=Server(('127.0.0.1',a.port),study,base/'WEB')
    except OSError:raise SystemExit('Port is occupied. Use the already running copy or choose another port; no process was stopped.')
    url='http://127.0.0.1:'+str(server.server_port)+'/'
    print(url+' | '+('COLLECTION OPEN' if a.collect else 'READ-ONLY PREVIEW')+' | Keep this terminal open; Ctrl+C stops the app.',flush=True)
    if not a.no_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()

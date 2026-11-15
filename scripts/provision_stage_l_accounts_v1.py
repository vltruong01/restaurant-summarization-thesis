"""One-time localhost account setup; secrets go to an interactive terminal only.

Existing users are never updated/reset. Store.add_user supplies the application's
unchanged salted PBKDF2 password hashing. No TEST reference action is issued.
"""
import argparse
import http.cookiejar
import json
from pathlib import Path
import secrets
import sqlite3
import sys
import urllib.request
from urllib.parse import urlsplit
from stage_l_reference_app_v1 import Store, need, new_json, now

ACCOUNTS=(('author01','author'),('checker01','checker'))
# A vocabulary is not a saved credential. Each actual phrase is generated anew
# with secrets.choice, held in memory, and never written to any file.
WORDS=tuple('''apple apricot atlas autumn bamboo banana basket beach beacon berry
birch blossom blue boat book bread breeze bridge brook bubble butter cabin
cactus candle canyon carrot castle cedar cherry chestnut circle cloud clover
cobalt cocoa coconut coral cotton creek crystal daisy dawn desert dolphin
dragon dream drift eagle earth echo elm ember emerald falcon feather fern field
fig finch firefly forest fox garden ginger glacier golden grape green grove
harbor hazel heron hill honey horizon island ivory jasmine jazz juniper kayak
kiwi kite lagoon lake lantern lavender lemon lilac lime linen lion lotus maple
marble meadow melon mint morning moss mountain music nectar ocean olive orange
orchid otter owl panda paper peach pearl pebble pecan pepper piano pine plum
pond poppy prairie purple quartz rabbit rainbow raven reef river robin rocket
rose ruby saddle sage sail salmon sand satin scarlet sea shell silver sky snow
sparrow spring spruce star stone storm strawberry stream summer summit sunset
sunrise swan tea timber toast trail tulip turtle valley velvet violet walnut
wave wheat willow wind winter wood yellow zebra'''.split())

def memorable_passphrase():
    return '-'.join(secrets.choice(WORDS).capitalize() for _ in range(6))

def user_rows(store):
    with store.connect() as c:
        return {r['id']:dict(r) for r in c.execute('SELECT * FROM users')}

def http_login_check(base_url,identity,password,role):
    url=urlsplit(base_url)
    need(url.scheme=='http' and url.hostname=='127.0.0.1' and url.port is not None
        and not url.username and not url.password and url.path in ['', '/'] and not url.query and not url.fragment,
        'Login verification must target explicit http://127.0.0.1:PORT only')
    base_url=base_url.rstrip('/')
    jar=http.cookiejar.CookieJar();client=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    request=urllib.request.Request(base_url+'/api/login',data=json.dumps({'identity':identity,'password':password}).encode('utf8'),
        headers={'Content-Type':'application/json','Origin':base_url})
    with client.open(request,timeout=15) as r:session=json.load(r)
    need(session['actor']=={'id':identity,'role':role},'HTTP login returned a different identity/role')
    try:
        with client.open(base_url+'/api/session',timeout=15) as r:check=json.load(r)
        need(check['actor']==session['actor'],'Session identity/role differs')
        with client.open(base_url+'/api/cases',timeout=15) as r:cases=json.load(r)
        return {'http_login':'PASS','http_session_role':role,'source_cases_accessible':len(cases['cases'])}
    finally:
        # End this technical login; do not leave a participant session open.
        request=urllib.request.Request(base_url+'/api/logout',data=b'{}',headers={
            'Content-Type':'application/json','Origin':base_url,'X-CSRF-Token':session['csrf']})
        with client.open(request,timeout=15) as r:need(json.load(r)['ok'],'Technical session logout failed')

def matches_account(store,identity,password):
    try:store.login(identity,password);return True
    except ValueError:return False

def provision(store,emit,verify_url=None,phrase_factory=memorable_passphrase):
    before=store.audit();existing=user_rows(store)
    # Preflight ALL requested names before creating anything. A conflicting role
    # needs explicit operator resolution, never an automatic permission change.
    for identity,role in ACCOUNTS:
        need(identity not in existing or existing[identity]['role']==role,
            'Existing account '+identity+' has a different role; unchanged. Explicit approval is required for a role change.')
    generated=[];results=[]
    for identity,role in ACCOUNTS:
        if identity in existing:
            emit(f'{identity} | {role} | EXISTS: account and password preserved; no password is retrievable or reprinted.')
            results.append({'username':identity,'role':role,'result':'PRESERVED_EXISTING_ACCOUNT','password_reset':False})
            continue
        password=None
        for _ in range(100):
            candidate=phrase_factory()
            if len(candidate)<12 or candidate in generated:continue
            # Also avoid choosing the same phrase as a preserved requested user,
            # without retrieving or changing their existing password.
            if any(matches_account(store,other,candidate) for other,_ in ACCOUNTS if other in existing):continue
            password=candidate;break
        need(password is not None,'Could not generate a distinct memorable passphrase')
        try:store.add_user(identity,role,password)
        except sqlite3.IntegrityError:
            current=user_rows(store)
            need(identity in current and current[identity]['role']==role,'Concurrent account role conflict; no existing row was changed')
            emit(f'{identity} | {role} | EXISTS: concurrent creation preserved; no password reset.')
            results.append({'username':identity,'role':role,'result':'PRESERVED_CONCURRENT_ACCOUNT','password_reset':False})
            continue
        generated.append(password)
        # Print once, immediately after the committed hash insert. No transcript,
        # report, JSON object or archive receives this plaintext credential.
        emit(f'\nUSERNAME: {identity}\nROLE:     {role}\nPASSWORD: {password}\n')
        need(store.login(identity,password)=={'id':identity,'role':role},'Stored password/role verification failed')
        checks={'stored_hash_login':'PASS'}
        if verify_url:checks.update(http_login_check(verify_url,identity,password,role))
        results.append({'username':identity,'role':role,'result':'CREATED','password_reset':False,**checks})
    after=store.audit()
    need(before==after,'Reference state changed during provisioning')
    current=user_rows(store)
    need(all(current[k]==v for k,v in existing.items()),'An existing user record changed')
    return {'status':'PASS_ACCOUNT_PROVISIONING','at_utc':now(),'accounts':results,
        'existing_users_unchanged':True,'reference_state_unchanged':True,
        'reference_progress':after['counts'],'plaintext_passwords_saved_to_file':False,
        'hashing':'Existing Store.add_user: salted PBKDF2-HMAC-SHA256, 200000 iterations',
        'independence_note':'Accounts must be assigned to two genuinely different eligible people; provisioning is not authorship, checking or approval.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--report',type=Path);p.add_argument('--verify-url')
    a=p.parse_args()
    need(sys.stdout.isatty(),'Run directly in an interactive terminal. Refusing to print credentials to a redirected file/log.')
    need(not a.report or not a.report.exists(),'Report must be a new file; existing report/account state will not be overwritten')
    print('Stage L account setup — record the newly displayed passphrases privately.\nExisting passwords are never reset. No human-reference action is performed.',flush=True)
    result=provision(Store(a.workspace),lambda text:print(text,flush=True),a.verify_url)
    if a.report:new_json(a.report,result)
    print('PASS: accounts ready; reference state unchanged. No plaintext password file was created.',flush=True)

if __name__=='__main__':main()

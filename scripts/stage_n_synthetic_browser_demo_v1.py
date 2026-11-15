"""Disposable browser smoke fixture: never uses a real study workspace."""
from pathlib import Path
import secrets
import tempfile
from test_stage_n_human_evaluation_v1 import fixture, enroll, HERE
from stage_n_human_evaluation_v1 import Server

if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='stage_n_BROWSER_SYNTHETIC_') as tmp:
        s=fixture(Path(tmp)/'synthetic');enroll(s)
        password=secrets.token_urlsafe(14);s.add_user('A01','rater',password)
        print('SYNTHETIC ONLY: http://127.0.0.1:8767 account A01 temporary password '+password,flush=True)
        server=Server(('127.0.0.1',8767),s,HERE/'stage_n_human_web_v1')
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:server.server_close()

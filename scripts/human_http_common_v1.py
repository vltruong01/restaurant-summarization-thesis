"""Small extracted common HTTP helpers; same behavior as the previous local app."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import datetime,json
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def need(ok,message):
    if not ok:raise ValueError(message)
def json_bytes(x):return (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode('utf8')
def new_json(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def safe_under(root,name):
    root=Path(root).resolve();p=(root/name).resolve();need(p.is_relative_to(root),'Path outside workspace');return p
class Server(ThreadingHTTPServer):daemon_threads=True
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def respond(self,status,data,ctype='application/json; charset=utf-8',cookie=None):
        raw=json_bytes(data) if ctype.startswith('application/json') else data
        self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if cookie:self.send_header('Set-Cookie',cookie)
        self.end_headers();self.wfile.write(raw)

"""Short-lived, loopback-only read-only preview; no human actions or labels."""
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from functools import partial
import time

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'outputs/stage_f/ai_assisted_calibration_v1'
    server=HTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(root)))
    server.timeout=1
    print('http://127.0.0.1:'+str(server.server_port)+'/human_check.html',flush=True)
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:server.handle_request()
    server.server_close()

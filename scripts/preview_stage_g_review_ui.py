"""Temporary loopback-only preview of the offline viewer; no directory or data writes."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / 'outputs/stage_g/pilot_review_ui_v1/dist/index.html'


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split('?',1)[0] not in {'/', '/index.html'}:
            self.send_error(404)
            return
        body = PAGE.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    assert PAGE.is_file()
    server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
    print('http://127.0.0.1:' + str(server.server_port) + '/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

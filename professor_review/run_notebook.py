"""Replay local analyses into a NEW output folder; leave the delivery snapshot intact."""
from pathlib import Path
import argparse
import json
import os
import re
import html
from urllib.parse import unquote
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New folder for reproduced notebook/HTML')
    args=parser.parse_args()
    base=Path(__file__).resolve().parent
    out=args.output.resolve()
    assert not out.exists(), 'Choose a new output folder.'
    assert not out.is_relative_to(base), 'Write reproduced results outside the delivered package.'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.dont_write_bytecode=True
    with tempfile.TemporaryDirectory(prefix='thesis_notebook_kernel_') as tmp:
        temp=Path(tmp);spec=temp/'kernels/thesis-summary'
        spec.mkdir(parents=True)
        (spec/'kernel.json').write_text(json.dumps({
            'argv':[sys.executable,'-X','utf8','-B','-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'Thesis CPU summary','language':'python',
            'env':{'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(temp/'matplotlib'),
                   'IPYTHONDIR':str(temp/'ipython'),
                   'MPLBACKEND':'module://matplotlib_inline.backend_inline'}}),encoding='utf-8')
        km=KernelManager(kernel_name='thesis-summary',
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temp/'kernels')]))
        notebook=nbformat.read(base/'THESIS_RESEARCH_OVERVIEW.ipynb',as_version=4)
        try:
            NotebookClient(notebook,km=km,resources={'metadata':{'path':str(base)}},timeout=180).execute()
        finally:
            if km.has_kernel:km.shutdown_kernel(now=True)
            km.cleanup_resources()
        out.mkdir(parents=True)
        nbformat.write(notebook,out/'THESIS_RESEARCH_OVERVIEW.ipynb')
        exporter=HTMLExporter(template_name='classic');exporter.mathjax_url=''
        body,_=exporter.from_notebook_node(notebook)
        # All outputs are static; remove optional CDN/widget scripts for offline reading.
        body=re.sub(r'<script\b[^>]*>.*?</script\s*>','',body,flags=re.I|re.S)
        body=re.sub(r'\bid="([^"]*)"',lambda m:'id="'+html.escape(unquote(html.unescape(html.unescape(m.group(1)))),quote=True)+'"',body)
        (out/'THESIS_RESEARCH_OVERVIEW.html').write_text(body,encoding='utf-8')
    print('PASS: local analyses reproduced. No GPU, network calls or model execution. Original snapshot unchanged.')
    print('Outputs:',out)
    print('For another replay, use the original package and this script; the reproduced pair uses its bundled inputs there.')

if __name__=='__main__':main()

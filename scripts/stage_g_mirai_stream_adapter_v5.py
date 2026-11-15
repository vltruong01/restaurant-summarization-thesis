"""Provider SSE envelope compatibility; retain exact text and never invent model metadata."""
import gzip
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from stage_g_mirai_stream_adapter_v4 import StreamProjection as BaseProjection
from prepare_stage_g_teacher_pilot import canonical
from run_stage_g_teacher_pilot_v4_1_recovery import now

CONTEXT=threading.local()
MODEL_HEADERS=('openai-model','x-model','x-model-id','x-resolved-model')


class StreamProjection(BaseProjection):
    def consume(self,event):
        if event.get('type')=='response.output_item.done':
            index=event.get('output_index',0)
            if index in self.item_done and self.item_done[index]!=event.get('item'):
                raise ValueError('Conflicting completed message items')
        super().consume(event)

    def project(self,completed):
        if completed.get('output') and not (self.parts or self.done or self.item_done):
            return completed,{'reconstructed_output':False,'model_metadata_added':False,
                              'SSE_text_consistency_checked':False,'text_origin':'complete_provider_response'}
        for slot in self.parts:
            if slot not in self.done:
                item=self.item_done.get(slot[0],{})
                content=item.get('content',[])
                if slot[1]>=len(content) or content[slot[1]].get('type')!='output_text':
                    raise ValueError('Text deltas have no completed text/item evidence')
                if content[slot[1]].get('text')!=''.join(self.parts[slot]):
                    raise ValueError('Completed message disagrees with text deltas')
        # Run consistency checks even when the final response already carries output.
        reconstructed,lineage=super().project({k:v for k,v in completed.items() if k!='output'})
        if completed.get('output'):
            def texts(output):
                return [(i,c,p.get('text')) for i,item in enumerate(output)
                    for c,p in enumerate(item.get('content',[])) if p.get('type')=='output_text']
            if texts(completed['output'])!=texts(reconstructed['output']):
                raise ValueError('Completed response disagrees with exact SSE text')
            return completed,{'reconstructed_output':False,'model_metadata_added':False,
                              'SSE_text_consistency_checked':True}
        assert reconstructed.get('model')==completed.get('model')
        lineage['SSE_text_consistency_checked']=True
        return reconstructed,lineage


def set_context(directory,jid):
    CONTEXT.directory=Path(directory);CONTEXT.jid=jid


def data_frames(stream,key,archive,started):
    """Archive original wire lines; parse the SSE data field without changing its text."""
    parts=[]
    while True:
        remaining=180-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('Teacher request exceeded frozen 180-second deadline')
        if getattr(stream,'fp',None) is None:break
        stream.fp.raw._sock.settimeout(min(120,remaining))
        line=stream.readline()
        if not line:break
        if key.encode() in line:raise ValueError('Provider echoed credential; reject without storing secret')
        archive.write(line)
        if time.monotonic()-started>180:raise TimeoutError('Teacher request exceeded frozen 180-second deadline')
        decoded=line.decode('utf-8').rstrip('\r\n')
        if not decoded:
            if parts:yield '\n'.join(parts);parts=[]
        elif decoded.startswith('data:'):
            value=decoded[5:]
            parts.append(value[1:] if value.startswith(' ') else value)
    if parts:yield '\n'.join(parts)


def call(base,key,body):
    directory=CONTEXT.directory;jid=CONTEXT.jid;directory.mkdir(parents=True,exist_ok=True)
    events=directory/(jid+'.sse.txt.gz');diagnostic=directory/(jid+'.json')
    assert not events.exists() and not diagnostic.exists(),'No automatic overwrite or resend'
    projection=StreamProjection();started=time.monotonic()
    report={'started_utc':now(),'requested_model':body['model'],'transport_version':'mirai_stream_adapter_v5',
        'overall_deadline_seconds':180,'socket_idle_timeout_seconds':120,'automatic_retries':0,
        'event_counts':{},'events_file':events.name,'provider_model_observations':[]}
    def safe(value):
        text=str(value).replace(key,'[REDACTED_TEACHER_CREDENTIAL]')
        text=re.sub(r'sk-[A-Za-z0-9_-]{20,}','[REDACTED_KEY_LIKE_TEXT]',text)
        return re.sub(r'(?i)Bearer\s+\S+','Bearer [REDACTED]',text)
    def observe(source,value):
        if isinstance(value,str) and value.strip():
            item={'source':source,'value':safe(value)}
            if item not in report['provider_model_observations']:report['provider_model_observations'].append(item)
    try:
        req=urllib.request.Request(base+'/responses',data=canonical(body),method='POST',
            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json',
                     'Accept':'text/event-stream','User-Agent':'codex-cli/0.149.0'})
        with events.open('xb') as file,gzip.GzipFile(fileobj=file,mode='wb',mtime=0) as archive:
            with urllib.request.urlopen(req,timeout=120) as stream:
                report['http_status']=getattr(stream,'status',None)
                report['request_ids']={h:safe(stream.headers.get(h)) for h in ('x-request-id','request-id','cf-ray') if stream.headers.get(h)}
                report['provider_model_headers']={h:safe(stream.headers.get(h)) for h in MODEL_HEADERS if stream.headers.get(h)}
                for h,value in report['provider_model_headers'].items():observe('HTTP.header.'+h,value)
                for data in data_frames(stream,key,archive,started):
                    if data=='[DONE]':break
                    event=json.loads(data);kind=event.get('type')
                    report['event_counts'][str(kind)]=report['event_counts'].get(str(kind),0)+1
                    response=event.get('response')
                    if isinstance(response,dict):observe(str(kind)+'.response.model',response.get('model'))
                    if kind in ('response.created','response.in_progress','response.completed'):
                        observe(str(kind)+'.model',event.get('model'))
                    projection.consume(event)
                    if kind=='response.completed':
                        assert isinstance(response,dict)
                        wire=directory/(jid+'.completed_wire.json')
                        with wire.open('x',encoding='utf-8') as handle:handle.write(json.dumps(response,ensure_ascii=False,indent=2)+'\n')
                        projected,lineage=projection.project(response)
                        report.update(status='COMPLETE_PROVIDER_STREAM_PROJECTED',response_id=response.get('id'),
                            response_model=response.get('model'),response_status=response.get('status'),
                            response_usage=response.get('usage'),output_projection=lineage)
                        return projected
                    if kind in ('error','response.failed','response.incomplete'):
                        report['failure_kind']=kind;report['provider_error']=safe(json.dumps(event,ensure_ascii=False))
                        raise RuntimeError('Provider stream failed; no automatic retry')
        report['failure_kind']='stream_EOF_without_completed_response'
        raise RuntimeError('No completed provider response; no candidate accepted')
    except Exception as error:
        report.update(status='FAILED_NO_AUTOMATIC_RETRY',error_type=type(error).__name__,error_message_redacted=safe(error))
        if isinstance(error,urllib.error.HTTPError):
            report['http_status']=error.code
            report['provider_http_error_body_redacted']=safe(error.read(65536).decode('utf-8',errors='replace'))
        raise
    finally:
        report.update(finished_utc=now(),elapsed_seconds=round(time.monotonic()-started,3))
        with diagnostic.open('x',encoding='utf-8') as handle:handle.write(safe(json.dumps(report,ensure_ascii=False,indent=2))+'\n')


def replay(events_path,completed):
    projection=StreamProjection();data=[];last=None
    def consume(value):
        nonlocal last
        if value=='[DONE]':return
        event=json.loads(value);projection.consume(event)
        if event.get('type')=='response.completed':last=event['response']
    with gzip.open(events_path,'rt',encoding='utf-8') as stream:
        for line in stream:
            line=line.rstrip('\r\n')
            if not line:
                if data:consume('\n'.join(data));data=[]
            elif line.startswith('data:'):
                value=line[5:];data.append(value[1:] if value.startswith(' ') else value)
    if data:consume('\n'.join(data))
    assert last==completed,'Completed wire response changed'
    return projection.project(last)

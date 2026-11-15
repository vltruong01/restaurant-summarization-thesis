"""Verbatim text views only; no scoring, model imports or factual labels."""
import re

def segments(text):
    result=[]
    for m in re.finditer(r'.+?(?:[.!?](?=\s|$)|$)',text,re.S):
        a,b=m.span()
        while a<b and text[a].isspace():a+=1
        while b>a and text[b-1].isspace():b-=1
        if a<b:result.append({'text':text[a:b],'start':a,'end':b})
    return result

def unique_segments(text):
    groups={}
    for s in segments(text):
        key=re.sub(r'\s+',' ',s['text']).lower()
        if key not in groups:groups[key]={'text':s['text'],'occurrences':[]}
        groups[key]['occurrences'].append({'start':s['start'],'end':s['end'],'verbatim':s['text']})
    return list(groups.values())

"""Single candidate serializer; exact originals and complete-source token accounting."""
from pathlib import Path
import hashlib
from tokenizers import Tokenizer
from stage_f_verification_workflow import read,sha

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['name','categories','city','state']
TEMPLATE={'status':'IMMUTABLE_AUDIT_CANDIDATE_NOT_PRODUCTION_D032_FREEZE',
          'business_header':'[BUSINESS_INFORMATION]','fields':FIELDS,'field_line':'{field}={exact_original_value}',
          'review_header':'[REVIEWS]','review_line':'[REV_{index}] stars={int_stars} date={original_date}\n{complete_original_text}',
          'visual_header':'[VISUAL_CONTEXT]','caption_marker':'[IMG_CAP_{index}]','caption_text':'stored_normalized_caption_unchanged',
          'section_separator':'\n\n','caption_separator':'\n\n','marker_body_separator':'\n',
          'extra_final_newline':False,'empty_visual_header_retained':True,'special_tokens':'once_per_complete_source',
          'markers_registered_as_special_tokens':False,'missing_metadata':'empty_string',
          'review_allocation':4000,'caption_increment_allowance':750,'complete_source_limit':5000,
          'caption_allowance_scope':'full_source_tokens_minus_same_business_reviews_only_base_tokens',
          'controlled_conditions':['reviews_only','unfiltered','filtered','shuffled'],
          'overflow_policy':'record_and_stop_zero_silent_truncation_or_removal'}


def text_sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def review_block(selected):
    assert selected==sorted(selected,key=lambda r:(r['date'],r['review_id']))
    return '\n\n'.join(f"[REV_{i}] stars={int(r['stars'])} date={r['date']}\n{r['text']}" for i,r in enumerate(selected,1))


def metadata_section(business):
    values=[]
    for field in FIELDS:
        value=business.get(field);value='' if value is None else value
        assert isinstance(value,str),field
        values.append(field+'='+value)
    return '[BUSINESS_INFORMATION]\n'+'\n'.join(values)


def base_source(business,block):return metadata_section(business)+'\n\n[REVIEWS]\n'+block


def visual_section(captions):
    assert all(isinstance(c,str) for c in captions)
    return '[VISUAL_CONTEXT]\n'+'\n\n'.join(f'[IMG_CAP_{i}]\n{c}' for i,c in enumerate(captions,1))


def assemble(base,captions=None):
    return base if captions is None else base+'\n\n'+visual_section(captions)


def pinned_tokenizer():
    manifest=read(ROOT/'assets/tokenizers/led-base-16384.manifest.json')
    assert manifest['model_id']=='allenai/led-base-16384'
    assert manifest['revision']=='38335783885b338d93791936c54bb4be46bebed9'
    folder=ROOT/manifest['relative_directory']
    for name,digest in manifest['sha256'].items():assert sha(folder/name)==digest
    tok=Tokenizer.from_file(str(folder/'tokenizer.json'));tok.no_padding();tok.no_truncation()
    return tok,manifest


def encoding(tok,text,special=True):
    e=tok.encode(text,add_special_tokens=special)
    if special:assert sum(e.special_tokens_mask)==2
    return e


def counts(tok,business,block,captions):
    base=base_source(business,block);source=assemble(base,captions)
    r=len(encoding(tok,block).ids);b=len(encoding(tok,base).ids);f=len(encoding(tok,source).ids)
    empty=len(encoding(tok,assemble(base,[])).ids)
    prefixes=[];previous=empty
    for n in range(1,len(captions)+1):
        current=len(encoding(tok,assemble(base,captions[:n])).ids)
        prefixes.append(current-previous);previous=current
    caption_increment=f-b
    assert caption_increment==empty-b+sum(prefixes) and f==r+(b-r)+caption_increment
    return {'review_block_tokens_with_special':r,'reviews_only_base_tokens':b,'full_source_tokens':f,
            'marginal_metadata_review_header_overhead':b-r,'caption_block_increment_tokens':caption_increment,
            'visual_header_separator_increment_tokens':empty-b,'complete_caption_prefix_increments':prefixes,
            'caption_body_standalone_sum':sum(len(encoding(tok,c,False).ids) for c in captions),
            'metadata_section_standalone_tokens':len(encoding(tok,metadata_section(business),False).ids),
            'visual_section_standalone_tokens':len(encoding(tok,visual_section(captions),False).ids),
            'caption_joint_boundary_delta':caption_increment-len(encoding(tok,visual_section(captions),False).ids),
            'special_tokens_added_per_complete_source':2,'caption_count':len(captions),
            'review_budget_overflow':r>4000,'caption_budget_overflow':caption_increment>750,'source_budget_overflow':f>5000,
            'declared_budget_envelope_tokens':4000+(b-r)+750,
            'declared_envelope_overflow':4000+(b-r)+750>5000,
            'base_sha256':text_sha(base),'source_sha256':text_sha(source),'review_block_sha256':text_sha(block)},source

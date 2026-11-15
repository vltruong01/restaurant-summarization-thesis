"""Future, explicitly approved epoch-1 sanity. Importing executes no model code."""
import json
from pathlib import Path
from led_large_decoder_label_view_v1 import require,remove_initial_bos,sanity_result,sha

def run_early_sanity(model,folder,approval,config,memory,root,binding,write,check_deadline):
    import torch
    from transformers import LEDTokenizerFast,GenerationConfig
    from peft import get_peft_model_state_dict
    from led_large_dynamic_amp_v2 import capture_rng,restore_rng
    from run_stage_i_d050_feasibility_v1 import fingerprint
    from led_large_decoder_label_view_v1 import CorrectedLabelCollator
    require(approval.get('early_sanity_generation_approved') is True,'Early generation not approved')
    require(approval.get('early_sanity_outputs_max')==4,'Wrong early sanity budget')
    path=Path(folder)/'early_sanity';path.mkdir(exist_ok=False)
    checkpoint=Path(folder)/'checkpoint-250'
    require((checkpoint/'checkpoint_integrity.json').is_file(),'Epoch1 checkpoint receipt required')
    all_cases=[json.loads(x) for x in (Path(root)/config['early_sanity']['cases_file']).read_text(encoding='utf-8').splitlines()]
    cases={c['business_id']:c for c in all_cases}
    tok=LEDTokenizerFast.from_pretrained(Path(root)/config['source_tokenizer'],local_files_only=True)
    g=GenerationConfig(**json.loads((Path(root)/config['effective_generation_config_file']).read_text(encoding='utf-8')))
    pack=CorrectedLabelCollator();rng=capture_rng()
    modes={name:m.training for name,m in model.named_modules()}
    old_cache=model.config.use_cache
    adapter={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    rows=[]
    try:
        model.eval()
        for bid in config['early_sanity']['business_ids']:
            check_deadline(approval)
            c=cases[bid]
            batch=pack([dict(input_ids=c['input_ids'],labels=remove_initial_bos(c['label_ids']))])
            batch={k:v.cuda() for k,v in batch.items() if k!='labels'}
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                ids=model.generate(**batch,generation_config=g)[0].cpu().tolist()
            check_deadline(approval)
            text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            record=dict(business_id=bid,subset=c['subset'],epoch=1,step=250,token_ids=ids,text=text,
                source_sha256=c['source_sha256'],checkpoint_sha256=sha(checkpoint/'adapter_model.safetensors'),
                technical_screen=sanity_result(ids,text),researcher_quality_status='NOT_REVIEWED')
            rows.append(record);write(path/(bid+'.json'),record);memory.check()
        require(all(torch.equal(v.detach().cpu(),adapter[k]) for k,v in get_peft_model_state_dict(model).items()),'Sanity modified adapters')
    finally:
        for name,m in model.named_modules():m.training=modes[name]
        model.config.use_cache=old_cache
        restore_rng(rng)
    require(all(m.training==modes[name] for name,m in model.named_modules()),'Training mode not restored')
    require(fingerprint(capture_rng())==fingerprint(rng),'RNG not restored after early sanity')
    report=dict(status='PASS_TECHNICAL_ONLY' if all(r['technical_screen']['status']=='PASS_TECHNICAL_ONLY' for r in rows) else 'FAIL_STOP',
        binding=binding,generation_calls=len(rows),epoch=1,step=250,rows=rows,
        RNG_restored=True,module_modes_restored=True,adapter_unchanged=True,quality_GO=False)
    write(path/'REPORT.json',report)
    require(report['status']=='PASS_TECHNICAL_ONLY','Early generation unhealthy; STOP without retry or setting changes')
    return report

"""Pinned greedy generative baseline; discrete choice, no invented confidence."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

from systematic.confirmation_runtime import ROOT, digest, failure, preflight_cases, prepare_run, save_runtime, validate

REPO='Qwen/Qwen2.5-7B-Instruct'
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
SYSTEM='You are a classifier. Follow the task and option descriptions. Output only the exact key of one offered option. Do not explain.'


def messages(case,request):
    prompt=json.dumps({'task':request['instruction'],'text':case['text'],'options':request['criteria']},ensure_ascii=False)
    return [{'role':'system','content':SYSTEM},{'role':'user','content':prompt}]


def parse_choice(raw,criteria):
    value=raw.strip()
    if value in criteria:
        return value
    # An exact JSON string key is a documented output-format tolerance.
    try:
        decoded=json.loads(value)
        if isinstance(decoded,str) and decoded in criteria:
            return decoded
    except ValueError:
        pass
    raise ValueError('Generated output is not an exact offered key')


def run(args):
    os.environ['USE_TF']='0'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    import transformers.utils
    import transformers.utils.import_utils
    transformers.utils.is_flash_attn_2_available=lambda:False
    transformers.utils.import_utils.is_flash_attn_2_available=lambda:False
    from transformers import AutoTokenizer,AutoModelForCausalLM
    assert torch.cuda.get_device_capability(0)[0] >= 8, 'Pinned BF16 baseline requires native Ampere-or-newer GPU'
    model_spec={'provider':'local-generative','repo':REPO,'revision':REVISION,'dtype':'bfloat16','score_track':'discrete'}
    output=Path(args.output_dir)
    if (output/'runtime.json').exists():
        previous=json.loads((output/'runtime.json').read_text(encoding='utf-8'))
        assert previous.get('gpu_name')==torch.cuda.get_device_name(0), 'Do not mix GPU families within a resumed baseline'
    manifest,cases,requests,existing=prepare_run(args.inputs,output,model_spec)
    pending=[r for r in requests.values() if r['request_id'] not in existing]
    if args.preflight:
        selected=preflight_cases(cases)
        pending=[r for r in pending if r['case_id'] in selected]
    if not pending:
        print(f'No pending responses ({len(existing)}/{len(requests)})');return
    torch.set_num_threads(1)
    cache=Path(os.environ.get('DECISION_BENCHMARK_CACHE', str(Path.cwd()/'cache')))/'huggingface'
    tok=AutoTokenizer.from_pretrained(REPO,revision=REVISION,cache_dir=cache,local_files_only=True,padding_side='left')
    tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(REPO,revision=REVISION,cache_dir=cache,local_files_only=True,
                                             torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    settings={'batch_size':args.batch_size,'do_sample':False,'max_new_tokens':32,'max_input_tokens':2048,
              'system_prompt':SYSTEM,'output_parser':'exact offered key or exact JSON string key; no other repair'}
    save_runtime(output,{'model':model_spec,'settings':settings,
                        'code_sha256':{name:digest(ROOT/'systematic'/name) for name in ('run_confirmation_llm.py','confirmation_runtime.py')},
                        'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers')},
                        'gpu_name':torch.cuda.get_device_name(0)})
    started,added= time.perf_counter(),0
    audit={'max_prompt_tokens':0,'prompt_over_budget':0,'output_truncated':0,'format_failure':0}
    with (output/'predictions.jsonl').open('a',encoding='utf-8') as stream,torch.inference_mode():
        for offset in range(0,len(pending),args.batch_size):
            batch=pending[offset:offset+args.batch_size]
            texts=[tok.apply_chat_template(messages(cases[r['case_id']],r),tokenize=False,add_generation_prompt=True) for r in batch]
            lengths=[len(tok(text,add_special_tokens=False)['input_ids']) for text in texts]
            audit['max_prompt_tokens']=max(audit['max_prompt_tokens'],max(lengths))
            valid=[i for i,n in enumerate(lengths) if n<=2048]
            saved={i:failure(batch[i],ValueError('Input exceeds frozen 2048-token budget'),'truncation_failure') for i,n in enumerate(lengths) if n>2048}
            audit['prompt_over_budget']+=len(saved)
            if valid:
                try:
                    encoded=tok([texts[i] for i in valid],add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
                    t0=time.perf_counter()
                    generated=model.generate(**encoded,do_sample=False,max_new_tokens=32,pad_token_id=tok.pad_token_id)
                    seconds=time.perf_counter()-t0
                    continuations=generated[:,encoded['input_ids'].shape[1]:]
                    eos=model.generation_config.eos_token_id
                    eos_ids={eos} if isinstance(eos,int) else set(eos)
                    for i,continuation in zip(valid,continuations):
                        raw=tok.decode(continuation,skip_special_tokens=True)
                        request=batch[i]
                        tokens=continuation.tolist()
                        truncated=not any(token in eos_ids for token in tokens)
                        try:
                            if truncated:
                                raise ValueError('Generated answer hit max_new_tokens without stop token')
                            choice=parse_choice(raw,request['criteria'])
                            p={'request_id':request['request_id'],'status':'ok','choice':choice,'probabilities':None,
                               'raw_text':raw,'prompt_tokens':lengths[i],'generated_tokens':next((j+1 for j,t in enumerate(tokens) if t in eos_ids),len(tokens)),
                               'batch_seconds':seconds,'batch_n':len(valid)}
                            validate(p,request)
                        except ValueError as exc:
                            p=failure(request,exc,'truncation_failure' if truncated else 'interface_failure')
                            p['raw_text']=raw;p['prompt_tokens']=lengths[i]
                            audit['output_truncated' if truncated else 'format_failure']+=1
                        saved[i]=p
                except Exception as exc:
                    for i in valid:
                        saved[i]=failure(batch[i],exc)
            for i in range(len(batch)):
                stream.write(json.dumps(saved[i],ensure_ascii=False)+'\n');added+=1
            stream.flush()
            if offset%(args.batch_size*25)==0 or offset+len(batch)==len(pending):
                print(f'Qwen saved {len(existing)+added}/{len(requests)}; {time.perf_counter()-started:.1f}s',flush=True)
    (output/f'token_audit_segment_{len(existing)}.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(f'Finished Qwen {len(existing)+added}/{len(requests)}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--preflight',action='store_true')
    run(p.parse_args())

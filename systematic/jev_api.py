"""Version-pinned TypeSafe HTTP client; credentials never enter artifacts."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid
import requests

ROOT=Path(__file__).resolve().parents[1]
MODEL='jev-1.13.0'
ENDPOINT='https://api.typesafe.ai/v1/systemone'
USD_PER_MILLION_INPUT=0.042  # official models page checked 2026-10-02

def api_key():
    key=os.environ.get('TYPESAFE_API_KEY')
    if not key:
        path=ROOT/'.private/typesafe.env'
        for line in path.read_text(encoding='utf-8').splitlines() if path.exists() else []:
            if line.startswith('TYPESAFE_API_KEY='):
                key=line.split('=',1)[1].strip()
    if not key:raise RuntimeError('Missing private TYPESAFE_API_KEY')
    return key

def validate_answer(answer, criteria, rounding_decimals=None):
    if answer.get('type')!='choice' or set(answer.get('probabilities',{}))!=set(criteria):
        raise ValueError('Choice type/probability keys do not match request')
    p=answer['probabilities']
    if any(type(v) not in [int,float] or not math.isfinite(v) or not 0<=v<=1 for v in p.values()):
        raise ValueError('Invalid probability values')
    tolerance=.001 if rounding_decimals is None else .5*10**(-rounding_decimals)*len(p)+1e-9
    if sum(p.values())<=0 or abs(sum(p.values())-1)>tolerance:
        raise ValueError(f'Probability sum outside tolerance: {sum(p.values()):.8f}')
    if answer.get('choice') not in criteria:
        raise ValueError('Chosen key absent from criteria')
    # Choice/probability argmax disagreement is measured as provider behavior,
    # not corrected or filtered out of the benchmark.
    c=answer.get('confidence')
    if type(c) not in [float,int] or not math.isfinite(c) or not 0<=c<=1:
        raise ValueError('Invalid native confidence')

class Client:
    def __init__(self,run_dir,budget_usd=1.0):
        self.run_dir=Path(run_dir)
        self.key=api_key()
        self.lock=threading.Lock()
        self.budget=budget_usd
        self.reserved=0
        self.used=0
        self.calls=0
        self.ledger=self.run_dir/'api_calls.jsonl'
        if self.ledger.exists():
            for line in self.ledger.read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                self.used+=row['accounted_input_tokens']
                self.calls+=1
    def models(self):
        r=requests.get('https://api.typesafe.ai/v1/models',headers={'Authorization':'Bearer '+self.key},timeout=30)
        if r.status_code!=200:raise RuntimeError(f'Model-list HTTP {r.status_code}')
        return [{'name':m['name'],'release_date':m.get('release_date')} for m in r.json()['models']]
    def evaluate(self,state,questions,purpose='inference'):
        payload={'state':state,'model':MODEL,'questions':questions}
        data=json.dumps(payload,ensure_ascii=False).encode('utf-8')
        # Byte-based conservative allowance, including per-question service overhead.
        allowance=len(data)+512*len(questions)
        with self.lock:
            if (self.used+self.reserved+allowance)*USD_PER_MILLION_INPUT/1e6>self.budget:
                raise RuntimeError('API budget guard reached')
            self.reserved+=allowance
        call_id=uuid.uuid4().hex
        start=time.perf_counter()
        status=None
        result=None
        native=None
        attempts=0
        error=None
        try:
            for attempt in range(4):
                attempts+=1
                r=requests.post(ENDPOINT,data=data,headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'},timeout=(15,90))
                status=r.status_code
                if status in [429,529] and attempt<3:
                    delay=min(20,max(1,float(r.headers.get('retry-after',2**attempt))))
                    time.sleep(delay)
                    continue
                if status!=200:raise RuntimeError(f'Evaluation HTTP {status}')
                result=r.json()
                break
            native={k:result[k] for k in ['model','answers','usage']}
            if self.key in json.dumps(native):raise ValueError('Unexpected credential in response')
            if result.get('model')!=MODEL:raise ValueError('Returned model version differs from pinned version')
            if set(result.get('answers',{}))!=set(questions):raise ValueError('Answer question IDs differ')
            for name,q in questions.items():validate_answer(result['answers'][name],q['criteria'],rounding_decimals=2)
            for token_name in ['input_tokens','output_tokens']:
                value=result.get('usage',{}).get(token_name)
                if type(value)!=int or value<0:raise ValueError('Invalid usage metadata')
            # Save only documented response fields, never headers or credentials.
            result=native
        except Exception as exc:
            # Avoid exception reprs containing Request objects/Authorization headers.
            error=type(exc).__name__ if not isinstance(exc,(ValueError,RuntimeError)) else str(exc).replace(self.key,'[redacted]')
        elapsed=time.perf_counter()-start
        usage=result.get('usage') if isinstance(result,dict) else None
        known=usage and type(usage.get('input_tokens'))==int and usage['input_tokens']>=0
        accounted=usage['input_tokens'] if known else allowance
        record={'call_id':call_id,'purpose':purpose,'model':MODEL,'http_status':status,'attempts':attempts,
            'utc':datetime.now(timezone.utc).isoformat(),'seconds':elapsed,'questions':len(questions),
            'usage':usage,'accounted_input_tokens':accounted,'usage_is_measured':bool(known),'error':error}
        with self.lock:
            self.reserved-=allowance
            self.used+=accounted
            self.calls+=1
            with self.ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            if native is not None and self.key not in json.dumps(native):
                with (self.run_dir/'api_responses.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'call_id':call_id,'purpose':purpose,'question_to_request':list(questions),'validation_error':error,'response':native},ensure_ascii=False)+'\n')
        if error:raise RuntimeError(error)
        return result,call_id,elapsed

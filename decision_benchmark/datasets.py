"""Pinned source download and exact frozen text reconstruction."""
import hashlib
import json
from pathlib import Path
import requests

from .artifacts import RESOURCES, read_json
from systematic.specs import TASKS

def source_path(key, spec, cache_dir, offline=False):
    task = key.split('/')[0]
    root = Path(cache_dir or (Path.cwd()/'cache/datasets')).resolve()
    path = root/task/spec['revision']/spec['filename']
    if not path.exists():
        if offline:
            raise FileNotFoundError(f'Pinned source not cached: {path}. Remove --offline to download it.')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.download')
        try:
            with requests.get(spec['url'], stream=True, timeout=(10,45)) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for block in response.iter_content(1024*1024):
                        if block: stream.write(block)
            if hashlib.sha256(temporary.read_bytes()).hexdigest()!=spec['sha256']:
                raise ValueError(f'Pinned source checksum mismatch: {key}')
            temporary.replace(path)
        finally:
            if temporary.exists(): temporary.unlink()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:
        raise ValueError(f'Cached source checksum mismatch: {key}')
    return path

def source_reader(task, source, path):
    if source['format']=='trec_raw':
        mapping = {name:i for i,(name,_) in enumerate(TASKS[task]['labels'])}
        values=[]
        for line in path.read_text(encoding='latin-1').splitlines():
            category,text=line.split(' ',1)
            values.append((text.strip(),mapping[category.split(':')[0]]))
        return lambda case: values[case['row_idx']]
    if source['format']=='clinc_json':
        values=json.loads(path.read_text(encoding='utf-8'))
        return lambda case: tuple(values[case['source_split']][case['row_idx']])
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError('Text reconstruction needs: pip install -e ".[data]"') from error
    columns=['label','title','content'] if task=='dbpedia' else ['label','text']
    table=pq.read_table(path, columns=columns)
    def get(case):
        index=case['row_idx']
        text=(table['title'][index].as_py()+'\n'+table['content'][index].as_py()) if task=='dbpedia' else table['text'][index].as_py()
        return text,table['label'][index].as_py()
    return get

def reconstruct(profile, cases, *, cache_dir=None, offline=False):
    release=read_json(RESOURCES/'release.json')
    readers={}
    output=[]
    order=read_json(RESOURCES/'frozen'/profile/'case_field_order.json')
    encoding=read_json(RESOURCES/'frozen'/profile/'case_encoding.json')
    for case in cases:
        task=case['task']
        key='clinc150/full' if task=='clinc150' else f"{task}/{case['source_split']}"
        if key not in readers:
            spec=release['sources'][key]
            readers[key]=source_reader(task,spec,source_path(key,spec,cache_dir,offline))
        text,label=readers[key](case)
        expected_label=case['label'] if task=='clinc150' else case['label_id']
        if label!=expected_label:
            raise ValueError(f"Source label mismatch for {case['case_id']}")
        if hashlib.sha256(text.encode()).hexdigest()!=case['text_sha256']:
            raise ValueError(f"Source text mismatch for {case['case_id']}")
        normalized=hashlib.sha256(' '.join(text.casefold().split()).encode()).hexdigest()
        if normalized!=case['normalized_text_sha256']:
            raise ValueError(f"Normalized text mismatch for {case['case_id']}")
        full={**case,'text':text}
        output.append({field:full[field] for field in order})
    return ''.join(json.dumps(case,ensure_ascii=encoding['ensure_ascii'])+encoding['line_ending'] for case in output).encode(encoding['encoding'])

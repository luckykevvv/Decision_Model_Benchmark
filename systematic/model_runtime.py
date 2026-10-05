from pathlib import Path
import os
import time
import json

ROOT = Path(__file__).resolve().parents[1]
REVISION = '55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851'

def load_agent(device='cpu'):
    cache = Path(os.environ.get('DECISION_BENCHMARK_CACHE', str(Path.cwd() / 'cache')))
    os.environ['USE_TF'] = '0'
    os.environ['HF_HOME'] = str(cache / 'hf_home')
    os.environ['HF_XET_CACHE'] = str(cache / 'xet')
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_CACHE'] = str(cache / 'huggingface')
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download('convaiinnovations/laya', revision=REVISION,
        cache_dir=os.environ['HF_HUB_CACHE'], allow_patterns=[
            'model.safetensors', 'rl_agent_config.json', 'encoder/*', 'tokenizer/*',
            'rl_common.py', 'rl_agent_api.py', 'email_utils.py'])
    import torch
    torch.set_num_threads(4)
    if os.environ.get('LAYA_DISABLE_FLASH_ATTN') == '1':
        # Laya already requests SDPA. Prevent Transformers importing an
        # optional FlashAttention binary incompatible with cluster GLIBC.
        import transformers.utils
        import transformers.utils.import_utils
        transformers.utils.is_flash_attn_2_available = lambda: False
        transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
    import laya
    return laya.load(snapshot, device=device), snapshot

if __name__ == '__main__':
    start = time.perf_counter()
    agent, snapshot = load_agent()
    question = {'topic': {'type': 'choice', 'instructions': 'Which topic best describes this news article?', 'criteria': {
        'World': 'World events and international politics', 'Sports': 'Sports and competitions',
        'Business': 'Finance and companies', 'Sci/Tech': 'Science and technology'}}}
    t0 = time.perf_counter()
    results = agent.predict_batch(['A new smartphone uses improved battery technology.'] * 8, question, batch_size=8)
    print(json.dumps({'load_seconds': t0-start, 'batch8_seconds': time.perf_counter()-t0, 'snapshot': snapshot, 'example': results[0]}, indent=2))

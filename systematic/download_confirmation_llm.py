"""Stage a pinned public checkpoint; no model load or inference on login node."""
from pathlib import Path
from huggingface_hub import snapshot_download
from systematic.run_confirmation_llm import REPO,REVISION

root=Path(__file__).resolve().parents[1]
path=snapshot_download(REPO,revision=REVISION,cache_dir=root/'cache/huggingface',
                       allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken'],max_workers=4)
print('Pinned checkpoint staged',path,flush=True)

from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('qwen/Qwen-1_8B-Chat', cache_dir='/data/checkpoints/Qwen-1_8B-Chat', revision='v1.0.0')

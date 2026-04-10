# # mkdir ./models/qwen2/Qwen2.5-0.5B/
# # python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-0.5B', local_dir='./models/qwen2/Qwen2.5-0.5B')"
# ln -sf ../configuration_qwen2.py ./models/qwen2/Qwen2.5-0.5B/configuration_qwen2.py
# ln -sf ../modeling_qwen2.py ./models/qwen2/Qwen2.5-0.5B/modeling_qwen2.py

# # mkdir ./models/qwen2/Qwen2.5-0.5B/
# # python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-0.5B', local_dir='./models/qwen2/Qwen2.5-0.5B')"
# ln -sf ../configuration_qwen2.py ./models/qwen2/Qwen2.5-1.5B/configuration_qwen2.py
# ln -sf ../modeling_qwen2.py ./models/qwen2/Qwen2.5-1.5B/modeling_qwen2.py

mkdir ./models/qwen2/Qwen2.5-7B/
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-7B', local_dir='./models/qwen2/Qwen2.5-7B')"
ln -sf ../configuration_qwen2.py ./models/qwen2/Qwen2.5-7B/configuration_qwen2.py
ln -sf ../modeling_qwen2.py ./models/qwen2/Qwen2.5-7B/modeling_qwen2.py
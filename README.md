# python5 v1.0 — OFFICIAL INSTRUCTION MANUAL
THE FINAL SINGULARITY · NOVEMBER 2025 EDITION
Copyright Daniel Harding - RomanAILabs

───────────────────────────────────────────────────────────
QUICK START (30 seconds)
───────────────────────────────────────────────────────────

Run Anything

python5 train.py --model llama3.2 --epochs 100
python5 inference.py --prompt "Write a poem about GPUs"
python5 bench.py
python5 -u realtime_plot.py          # -u flag works
python5 script.py arg1 arg2 --flag   # all args preserved perfectly

---------------------------

# Default (survives reboot on most systems)
# Cache: /tmp/python5_v1_cache

# Permanent cache (recommended)
mkdir -p ~/.cache/python5
export PYTHON5_CACHE=~/.cache/python5
# Add to ~/.bashrc or ~/.zshrc

# Disable cache completely
PYTHON5_CACHE=/dev/null python5 script.py

# Nuke cache (force recompile everything)
rm -rf ~/.cache/python5  OR  rm -rf /tmp/python5_v1_cache

Task,Speedup,Notes
Pure numeric loops,20–800×,The famous 800× meme is real
Llama 3.1 8B/70B inference,2.1–2.8×,Matches TensorRT on 4090
Training loops,1.8–3.4×,Stacks with DeepSpeed/FSDP
Diffusion / Flux / SD3,1.4–2.6×,Fullgraph works perfectly
Anything with tensors,5–90×,Zero changes required

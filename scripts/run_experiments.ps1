param(
    [string]$Model = "EleutherAI/pythia-70m",
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"

python scripts/run_ppl.py --model $Model --dataset builtin --max-samples 3 --device $Device --output results/ppl_builtin.json
python scripts/run_speed.py --model $Model --method dense --max-new-tokens 32 --device $Device --output results/speed_dense.json
python scripts/run_speed.py --model $Model --method window --cache-budget 64 --recent 32 --max-new-tokens 32 --device $Device --output results/speed_window.json
python scripts/run_speed.py --model $Model --method hybrid --cache-budget 64 --recent 32 --max-new-tokens 32 --device $Device --output results/speed_hybrid.json

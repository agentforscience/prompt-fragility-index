#!/bin/bash
# Throughput benchmark: bench.sh <model.gguf> <threads> <parallel>
cd "$(dirname "$0")/.."
source .venv/bin/activate
LS=tools/llamacpp/llama-b11624/llama-server
$LS -m "$1" --port 18391 -t $2 -np $3 -c $((2048*$3)) --no-webui > logs/llama_bench_t$2.log 2>&1 &
SP=$!
for i in $(seq 90); do curl -s localhost:18391/health | grep -q ok && break; sleep 2; done
timeout 300 python - "$3" "$2" << 'PY'
import time, sys, concurrent.futures as cf
from openai import OpenAI
c=OpenAI(base_url="http://localhost:18391/v1",api_key="x")
n=int(sys.argv[1])
def f(i):
    r=c.chat.completions.create(model="x",messages=[{"role":"user","content":f"Count from {i} upward, one number per line."}],max_tokens=64,temperature=0)
    return r.usage.completion_tokens
t=time.time()
with cf.ThreadPoolExecutor(n) as ex: res=list(ex.map(f,range(n)))
d=time.time()-t; print(f"threads={sys.argv[2]} par={n} {d:.1f}s {sum(res)} toks {sum(res)/d:.1f} tok/s", flush=True)
PY
kill $SP

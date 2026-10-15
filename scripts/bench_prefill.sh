#!/bin/bash
# Prefill-heavy benchmark: bench_prefill.sh <model.gguf> <threads> <parallel>
cd "$(dirname "$0")/.."
source .venv/bin/activate
LS=tools/llamacpp/llama-b11624/llama-server
$LS -m "$1" --port 18392 -t $2 -tb $2 -np $3 -c $((1024*$3)) -b 2048 -ub 512 --poll 0 --no-webui > logs/llama_benchpf_t$2.log 2>&1 &
SP=$!
for i in $(seq 90); do curl -s localhost:18392/health | grep -q ok && break; sleep 2; done
timeout 400 python - "$3" "$2" << 'PY'
import time, sys, json, concurrent.futures as cf
from openai import OpenAI
c=OpenAI(base_url="http://localhost:18392/v1",api_key="x")
n=int(sys.argv[1])
rows=[json.loads(l) for l in open("datasets/ag_news_test.jsonl")][:48]
inst="You are a news topic classifier. Read the news article below and decide which topic it belongs to. The possible topics are: World, Sports, Business, Sci/Tech. Respond with only the topic name and nothing else."
def f(r):
    x=c.chat.completions.create(model="x",messages=[{"role":"user","content":inst+"\n\nArticle: "+r["text"]*3+"\n\nTopic:"}],max_tokens=4,temperature=0)
    return x.usage.prompt_tokens, x.usage.completion_tokens
t=time.time()
with cf.ThreadPoolExecutor(n) as ex: res=list(ex.map(f,rows))
d=time.time()-t; p=sum(r[0] for r in res)
print(f"threads={sys.argv[2]} par={n} {d:.1f}s prompt_toks={p} -> {p/d:.0f} prompt tok/s, {len(rows)/d:.2f} req/s", flush=True)
PY
kill $SP

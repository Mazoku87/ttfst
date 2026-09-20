# ttfst — time to first speakable token

Streaming benchmarks publish **time to first token**. For anything that speaks, that number
is misleading, and the gap between it and the number a listener actually feels is where a
voice assistant goes wrong.

I found this the expensive way.

A turn in my voice assistant stalled for **48,347 ms**. I went looking for a bug in the
streaming path and there wasn't one. The model had emitted **1,283 reasoning tokens and an
empty response**. Time to first token was healthy the whole time, because reasoning tokens
are tokens. Nothing could be spoken, so the user heard silence.

Text-to-speech cannot start on a token either. It starts on a clause or a sentence, since
prosody needs to know where the phrase is going. So the number a listener waits for is the
time until **the first speakable sentence exists**, and that is what this measures.

## What it reports

| | |
|---|---|
| `TTFB` | connection open, first byte of the stream |
| `TTFT` | first token of any kind, reasoning included — **the published number** |
| `TTFC` | first token of user-visible content |
| `TTFST` | first complete sentence — **when audio can begin** |

`TTFT → TTFC` is thinking the user cannot hear. `TTFC → TTFST` is the tail of the first
sentence. On a reasoning model the first gap runs to tens of seconds while TTFT looks fine.

## Usage

No dependencies. Standard library only, so it runs wherever python does.

```bash
# any OpenAI-compatible endpoint
python ttfst.py --base-url http://localhost:11434/v1 --model qwen2.5:7b
python ttfst.py --base-url https://api.openai.com/v1 --model gpt-4o-mini --api-key $OPENAI_API_KEY
python ttfst.py --base-url http://localhost:8000/v1 --model my-model -n 5 --json
```

```
run     TTFB    TTFT    TTFC   TTFST   total  first sentence
1      0.04s   0.09s   3.71s   4.02s   6.88s  A vector database stores embeddings.
2      0.03s   0.08s   3.44s   3.79s   6.51s  A vector database stores embeddings.

worst time to speech 4.02s against a 5.0s budget -- OK
longest stretch where tokens flowed and nothing was speakable: 3.62s (1,462 chars of reasoning)
```

That last line is the one worth watching. It is silence the user experiences and no standard
benchmark reports.

## Details that turned out to matter

**Reasoning arrives in three different shapes.** Deepseek-style servers use
`reasoning_content`, some use `reasoning`, and others inline the model's own `<think>` tags
into `content`. All three are unspeakable and all three have to be recognised, or the
measurement is wrong in exactly the way this tool exists to catch. `<think>` blocks split
across chunk boundaries are handled, because they do split.

**An abbreviation is not a sentence end.** Flushing `Dr.` to TTS produces a clipped
half-word, which is worse than waiting. Same for the decimal point in `3.50`. Colons and
semicolons do count, since they give a speakable clause and streaming TTS flushes on them.

**A reply that never terminates is still speakable once the stream closes.** Otherwise a
one-line answer with no full stop reports as infinite latency.

## Tests

```bash
python -m unittest discover -s tests -v
```

18 tests. The end-to-end one runs against a mock server that streams reasoning steadily and
produces no speakable content for a set interval — it asserts that TTFT looks healthy while
TTFST does not, which is the whole claim.

## Where this came from

Extracted from the instrumentation for a real-time voice assistant I built and run on a
single RTX 4070 laptop with 8GB of VRAM: voice activity detection, streaming speech
recognition, local inference, and streaming speech out, with a routing layer that sends each
request to a local model, a confidential GPU enclave, or a frontier API depending on cost,
latency, capability and privacy.

Pipelining a previously sequential turn loop in that system took time to first audio from
**2.13s to 0.65s**. The 48-second stall above is what taught me which number to optimise
first.

## License

MIT

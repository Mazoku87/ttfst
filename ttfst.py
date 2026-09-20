#!/usr/bin/env python3
"""ttfst -- time to first speakable token.

Streaming benchmarks publish time-to-first-token. For anything that speaks, that number is
misleading, and the gap between the two is where a voice assistant goes wrong.

I found this the expensive way. A turn in my voice assistant stalled for 48,347 ms and I went
looking for a bug in the streaming path. There wasn't one. The model had emitted 1,283
reasoning tokens and an empty response. Time-to-first-token was healthy the entire time,
because reasoning tokens are tokens. Nothing could be spoken, so the user heard silence.

Text-to-speech also cannot start on a token. It starts on a clause or a sentence, because
prosody needs to know where the phrase is going. So the number a listener actually waits for
is the time until the first SPEAKABLE SENTENCE exists, and that is what this measures.

Four numbers, against any OpenAI-compatible streaming endpoint:

  TTFB    connection open, first byte of the stream
  TTFT    first token of any kind, reasoning included -- the published number
  TTFC    first token of user-visible content
  TTFST   first complete sentence, which is when audio can begin

TTFT minus TTFC is thinking the user cannot hear. TTFC minus TTFST is the tail of the first
sentence. On a reasoning model the first gap can be tens of seconds while TTFT looks fine.

No dependencies. Standard library only, so it runs wherever python does.

  python ttfst.py --base-url http://localhost:11434/v1 --model qwen2.5:7b
  python ttfst.py --base-url https://api.openai.com/v1 --model gpt-4o-mini --api-key $OPENAI_API_KEY
  python ttfst.py --base-url http://localhost:8000/v1 --model my-model -n 5 --json
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

__version__ = '1.0.0'

# A sentence ends at . ! ? -- but not inside an abbreviation or a decimal, and TTS needs the
# following space to be sure the sentence is closed. A colon or a semicolon also gives a
# speakable clause, which is why streaming TTS implementations usually flush on them too.
_SENTENCE_END = re.compile(r'(?<![A-Z])[.!?:;](?=\s)|[.!?](?=$)')
_ABBREV = re.compile(r'\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|e\.g|i\.e|fig|no|approx)\.$', re.I)
_DECIMAL = re.compile(r'\d\.$')

# Reasoning arrives in one of three shapes depending on who served it.
_THINK_OPEN = re.compile(r'<(?:think|thinking|reasoning)>', re.I)
_THINK_CLOSE = re.compile(r'</(?:think|thinking|reasoning)>', re.I)


def first_sentence_end(text):
    """Index just past the first real sentence terminator, or -1.

    Returns a position, not a bool, so the caller can show what would have been spoken. An
    abbreviation or a decimal point is not a sentence end: flushing "Dr." to TTS produces a
    clipped half-word, which is worse than waiting.
    """
    for m in _SENTENCE_END.finditer(text):
        head = text[:m.end()]
        if _ABBREV.search(head) or _DECIMAL.search(head):
            continue
        return m.end()
    return -1


def extract_delta(obj):
    """(reasoning_text, content_text) from one SSE chunk, across the shapes in the wild.

    Deepseek-style servers put it in `reasoning_content`, some in `reasoning`, and others
    inline the model's own <think> tags into `content`. All three are reasoning and none of
    them can be spoken, so all three have to be recognised or the measurement is wrong in
    exactly the way this tool exists to catch.
    """
    try:
        delta = obj['choices'][0].get('delta') or {}
    except (KeyError, IndexError, TypeError):
        return '', ''
    reasoning = delta.get('reasoning_content') or delta.get('reasoning') or ''
    content = delta.get('content') or ''
    if isinstance(reasoning, list):
        reasoning = ''.join(str(p.get('text', '')) for p in reasoning if isinstance(p, dict))
    return str(reasoning or ''), str(content or '')


class Turn:
    """One streamed completion, timed. Separates thinking from speech as it arrives."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.ttfb = self.ttft = self.ttfc = self.ttfst = None
        self.reasoning_chars = 0
        self.content = ''
        self.first_sentence = ''
        self._in_think = False

    def _now(self):
        return time.perf_counter() - self.t0

    def saw_first_byte(self):
        if self.ttfb is None:
            self.ttfb = self._now()

    def feed(self, reasoning, content):
        """Absorb one chunk. Order matters: a token is thinking until proven speakable."""
        if reasoning:
            self.reasoning_chars += len(reasoning)
            if self.ttft is None:
                self.ttft = self._now()
        if not content:
            return
        if self.ttft is None:
            self.ttft = self._now()

        # Inline <think> blocks: everything between the tags is reasoning, and the text after
        # the closing tag is the first thing anyone could hear.
        while content:
            if self._in_think:
                m = _THINK_CLOSE.search(content)
                if not m:
                    self.reasoning_chars += len(content)
                    return
                self.reasoning_chars += m.start()
                content = content[m.end():]
                self._in_think = False
                continue
            m = _THINK_OPEN.search(content)
            if m:
                speakable, content = content[:m.start()], content[m.end():]
                self._in_think = True
            else:
                speakable, content = content, ''
            if not speakable:
                continue
            if self.ttfc is None and speakable.strip():
                self.ttfc = self._now()
            self.content += speakable
            if self.ttfst is None:
                end = first_sentence_end(self.content)
                if end > 0:
                    self.ttfst = self._now()
                    self.first_sentence = self.content[:end].strip()

    def finish(self):
        """A reply with no sentence terminator is still speakable once the stream closes."""
        if self.ttfst is None and self.content.strip():
            self.ttfst = self._now()
            self.first_sentence = self.content.strip()
        self.total = self._now()
        return self


def run_turn(base_url, model, prompt, api_key=None, timeout=180, max_tokens=None):
    body = {'model': model, 'stream': True,
            'messages': [{'role': 'user', 'content': prompt}]}
    if max_tokens:
        body['max_tokens'] = max_tokens
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    req = urllib.request.Request(base_url.rstrip('/') + '/chat/completions',
                                 data=json.dumps(body).encode(), headers=headers)
    turn = Turn()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            turn.saw_first_byte()
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            payload = line[5:].strip()
            if payload == '[DONE]':
                break
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            turn.feed(*extract_delta(obj))
    return turn.finish()


def fmt(v):
    return '  --  ' if v is None else '%6.2fs' % v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL',
                                                         'http://localhost:11434/v1'))
    ap.add_argument('--model', required=True)
    ap.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY'))
    ap.add_argument('--prompt', default='In two sentences, what is a vector database?')
    ap.add_argument('-n', '--runs', type=int, default=3)
    ap.add_argument('--max-tokens', type=int, default=None)
    ap.add_argument('--budget', type=float, default=5.0,
                    help='seconds a listener will tolerate before audio starts')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args(argv)

    turns = []
    for i in range(args.runs):
        try:
            turns.append(run_turn(args.base_url, args.model, args.prompt,
                                  args.api_key, max_tokens=args.max_tokens))
        except Exception as e:
            print('run %d failed: %s' % (i + 1, e), file=sys.stderr)
    if not turns:
        return 2

    if args.json:
        print(json.dumps([{'ttfb': t.ttfb, 'ttft': t.ttft, 'ttfc': t.ttfc,
                           'ttfst': t.ttfst, 'total': t.total,
                           'reasoning_chars': t.reasoning_chars,
                           'first_sentence': t.first_sentence} for t in turns], indent=2))
        return 0

    print('%-4s %7s %7s %7s %7s %7s  %s'
          % ('run', 'TTFB', 'TTFT', 'TTFC', 'TTFST', 'total', 'first sentence'))
    for i, t in enumerate(turns, 1):
        print('%-4d %7s %7s %7s %7s %7s  %s'
              % (i, fmt(t.ttfb), fmt(t.ttft), fmt(t.ttfc), fmt(t.ttfst), fmt(t.total),
                 (t.first_sentence[:46] or '(nothing speakable)')))

    spoken = [t.ttfst for t in turns if t.ttfst is not None]
    if spoken:
        worst = max(spoken)
        print('\nworst time to speech %.2fs against a %.1fs budget -- %s'
              % (worst, args.budget, 'OK' if worst <= args.budget else 'OVER'))
    silent = [t for t in turns if t.ttft is not None and t.ttfc is not None]
    if silent:
        gap = max(t.ttfc - t.ttft for t in silent)
        chars = max(t.reasoning_chars for t in turns)
        if gap > 0.05:
            print('longest stretch where tokens flowed and nothing was speakable: %.2fs'
                  ' (%d chars of reasoning)' % (gap, chars))
    return 0


if __name__ == '__main__':
    sys.exit(main())

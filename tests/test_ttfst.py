#!/usr/bin/env python3
"""Tests for ttfst, including an end-to-end run against a mock streaming server.

The mock reproduces the failure that motivated the tool: a model that streams reasoning
tokens steadily and produces no speakable content for a long time. Time-to-first-token looks
healthy throughout, and a listener hears nothing.
"""
import json
import os
import socket
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ttfst import Turn, extract_delta, first_sentence_end, run_turn  # noqa: E402


class SentenceBoundaries(unittest.TestCase):
    def test_plain_sentence(self):
        self.assertEqual(first_sentence_end('Hello there. And more'), 12)

    def test_no_terminator_yet(self):
        self.assertEqual(first_sentence_end('Hello there and more'), -1)

    def test_abbreviation_is_not_a_sentence_end(self):
        # Flushing "Dr." to TTS produces a clipped half-word.
        self.assertEqual(first_sentence_end('Dr. Bergan spoke'), -1)

    def test_decimal_is_not_a_sentence_end(self):
        self.assertEqual(first_sentence_end('It costs 3.50 dollars'), -1)

    def test_decimal_then_real_sentence(self):
        idx = first_sentence_end('It costs 3.50 dollars. Next')
        self.assertEqual('It costs 3.50 dollars.', 'It costs 3.50 dollars. Next'[:idx])

    def test_clause_terminator_counts(self):
        self.assertEqual(first_sentence_end('One thing: two'), 10)

    def test_terminator_at_end_of_buffer(self):
        self.assertEqual(first_sentence_end('Done.'), 5)


class DeltaShapes(unittest.TestCase):
    def test_reasoning_content_field(self):
        chunk = {'choices': [{'delta': {'reasoning_content': 'hmm'}}]}
        self.assertEqual(extract_delta(chunk), ('hmm', ''))

    def test_reasoning_field(self):
        chunk = {'choices': [{'delta': {'reasoning': 'hmm'}}]}
        self.assertEqual(extract_delta(chunk), ('hmm', ''))

    def test_content_field(self):
        chunk = {'choices': [{'delta': {'content': 'hi'}}]}
        self.assertEqual(extract_delta(chunk), ('', 'hi'))

    def test_empty_delta_is_survivable(self):
        self.assertEqual(extract_delta({'choices': [{'delta': {}}]}), ('', ''))

    def test_malformed_chunk_is_survivable(self):
        self.assertEqual(extract_delta({'nonsense': True}), ('', ''))


class ThinkTagsAreNotSpeakable(unittest.TestCase):
    def test_inline_think_block_is_reasoning(self):
        t = Turn()
        t.feed('', '<think>weighing options</think>')
        t.feed('', 'The answer is four.')
        t.finish()
        self.assertEqual(t.first_sentence, 'The answer is four.')
        self.assertGreater(t.reasoning_chars, 0)

    def test_think_block_split_across_chunks(self):
        t = Turn()
        t.feed('', '<think>part one ')
        t.feed('', 'part two</think>Hello.')
        t.finish()
        self.assertEqual(t.first_sentence, 'Hello.')

    def test_ttfc_precedes_ttfst_when_sentence_is_incomplete(self):
        t = Turn()
        t.feed('', 'A partial clause with no end yet')
        self.assertIsNotNone(t.ttfc)
        self.assertIsNone(t.ttfst)

    def test_unterminated_reply_is_still_speakable_at_close(self):
        t = Turn()
        t.feed('', 'no terminator here')
        t.finish()
        self.assertEqual(t.first_sentence, 'no terminator here')

    def test_reasoning_only_reply_is_never_speakable(self):
        t = Turn()
        t.feed('thinking hard', '')
        t.finish()
        self.assertIsNotNone(t.ttft)
        self.assertIsNone(t.ttfst)
        self.assertIsNone(t.ttfc)


def _sse(obj):
    return ('data: ' + json.dumps(obj) + '\n\n').encode()


class _MockHandler(BaseHTTPRequestHandler):
    """Streams reasoning for REASONING_SECS, then one real sentence."""
    REASONING_SECS = 0.6
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()
        deadline = time.time() + self.REASONING_SECS
        while time.time() < deadline:
            self._chunk(_sse({'choices': [{'delta': {'reasoning_content': 'step '}}]}))
            time.sleep(0.02)
        self._chunk(_sse({'choices': [{'delta': {'content': 'A vector database '}}]}))
        self._chunk(_sse({'choices': [{'delta': {'content': 'stores embeddings.'}}]}))
        self._chunk(b'data: [DONE]\n\n')
        self._chunk(b'')

    def _chunk(self, payload):
        self.wfile.write(('%X\r\n' % len(payload)).encode() + payload + b'\r\n')
        self.wfile.flush()


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s = socket.socket()
        s.bind(('127.0.0.1', 0))
        cls.port = s.getsockname()[1]
        s.close()
        cls.server = HTTPServer(('127.0.0.1', cls.port), _MockHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_ttft_looks_healthy_while_nothing_is_speakable(self):
        """The whole point. TTFT is fast, TTFST is not, and only TTFST matches the listener."""
        t = run_turn('http://127.0.0.1:%d/v1' % self.port, 'mock', 'hi')
        self.assertLess(t.ttft, 0.3, 'TTFT should look healthy')
        self.assertGreaterEqual(t.ttfst, _MockHandler.REASONING_SECS,
                                'TTFST must include the silent reasoning stretch')
        self.assertGreater(t.ttfst - t.ttft, 0.4, 'the gap is the thing being measured')
        self.assertEqual(t.first_sentence, 'A vector database stores embeddings.')
        self.assertGreater(t.reasoning_chars, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

# SPDX-License-Identifier: Apache-2.0
"""Tests for the prompt-side splice of re-sent sampled output tokens."""

import logging
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler


class _PieceTokenizer:
    """Fake byte-level tokenizer: each id is a byte piece, decoded as UTF-8
    with errors="replace" (like byte-level BPE).

    Ids without an entry spell ``<id>``, so they never collide. Equal-text
    pairs: [10, 11] and [12] spell "abc"; [20, 21] and [22, 23] spell "qrs";
    [40] * 9 and [41] spell nine "m". 30 and 31 are different invalid bytes.
    70-72 are the three bytes of "€". 99 is a special token spelling "c".
    """

    PIECES = {
        10: b"ab",
        11: b"c",
        12: b"abc",
        15: b"x",
        20: b"q",
        21: b"rs",
        22: b"qr",
        23: b"s",
        30: b"\xff",
        31: b"\xfe",
        40: b"m",
        41: b"m" * 9,
        70: b"\xe2",
        71: b"\x82",
        72: b"\xac",
        73: b"z",
        74: b"y",
        99: b"c",
    }

    def __init__(self, encode_map=None, special_ids=(99,)):
        self.encode_map = dict(encode_map or {})
        self.all_special_ids = list(special_ids)

    def decode(self, ids, **kwargs):
        raw = b"".join(self.PIECES.get(int(i), f"<{int(i)}>".encode()) for i in ids)
        return raw.decode("utf-8", errors="replace")

    def encode(self, text, add_special_tokens=True):
        return list(self.encode_map[text])


A = list(range(1000, 1100))  # shared prefix, 100 tokens
B = list(range(2000, 2050))  # shared text after the first divergence
D = list(range(4000, 4040))  # shared text after the second divergence
C = list(range(3000, 3030))  # new-turn tokens the stored sequence never saw

STORED = A + [10, 11] + B  # sampled split "ab" + "c"
PROMPT = A + [12] + B + C  # canonical "abc"
SPLICED = A + [10, 11] + B + C

STORED_2 = A + [10, 11] + B + [15] + D  # second divergence: "x" vs "q"
PROMPT_2 = A + [12] + B + [20] + D + C

STORED_M = A + [10, 11] + B + [22, 23] + D  # two re-alignable divergences
PROMPT_M = A + [12] + B + [20, 21] + D + C

# 70 historical splits (more than the old 64 cap), each growing the prompt by 1.
SEGS = [list(range(5000 + 3 * k, 5003 + 3 * k)) for k in range(70)]
STORED_MANY = A + [t for s in SEGS for t in [10, 11] + s]
PROMPT_MANY = A + [t for s in SEGS for t in [12] + s] + C


def _make_scheduler(block_size=64, encode_map=None):
    sched = object.__new__(Scheduler)
    sched._cache_probe_seqs = deque(maxlen=8)
    sched.tokenizer = _PieceTokenizer(encode_map=encode_map)
    sched.config = MagicMock(spec=[])
    sched.config.paged_cache_block_size = block_size
    sched.config.max_num_seqs = 4
    sched.block_aware_cache = object()
    sched._unreconstructible_cache_model = False
    return sched


class TestSpliceAgainstStored:
    def test_splices_sampled_split_back_into_prompt(self):
        sched = _make_scheduler()
        new, p0, aligned, resolved = sched._splice_against_stored(PROMPT, STORED)
        assert new == SPLICED
        assert (p0, aligned, resolved) == (100, len(STORED), 1)

    def test_splices_when_prompt_carries_the_longer_split(self):
        sched = _make_scheduler()
        stored = A + [12] + B
        prompt = A + [10, 11] + B + C
        new, p0, aligned, resolved = sched._splice_against_stored(prompt, stored)
        assert new == A + [12] + B + C
        assert (p0, aligned, resolved) == (100, len(stored), 1)

    def test_text_mismatch_is_noop(self):
        sched = _make_scheduler()
        stored = A + [15] + B  # "x" is not "abc"
        new, p0, aligned, resolved = sched._splice_against_stored(PROMPT, stored)
        assert new is PROMPT
        assert (p0, aligned, resolved) == (100, 100, 0)

    def test_identical_prefix_is_noop(self):
        sched = _make_scheduler()
        prompt = STORED + C
        new, p0, aligned, resolved = sched._splice_against_stored(prompt, STORED)
        assert new is prompt
        assert (p0, aligned, resolved) == (len(STORED), len(STORED), 0)

    def test_window_is_bounded(self):
        sched = _make_scheduler()
        stored = A + [40] * 9 + B  # nine "m" tokens
        prompt = A + [41] + B + C  # one "mmmmmmmmm" token
        new, _, _, resolved = sched._splice_against_stored(prompt, stored)
        assert new is prompt and resolved == 0
        new, _, aligned, resolved = sched._splice_against_stored(
            prompt, stored, window=9
        )
        assert new == A + [40] * 9 + B + C
        assert (aligned, resolved) == (len(stored), 1)

    def test_multiple_divergences_are_spliced(self):
        sched = _make_scheduler()
        new, p0, aligned, resolved = sched._splice_against_stored(PROMPT_M, STORED_M)
        assert new == A + [10, 11] + B + [22, 23] + D + C
        assert (p0, aligned, resolved) == (100, len(STORED_M), 2)

    def test_stops_at_first_unresolvable_divergence(self):
        sched = _make_scheduler()
        new, p0, aligned, resolved = sched._splice_against_stored(PROMPT_2, STORED_2)
        assert new == A + [10, 11] + B + [20] + D + C
        assert (p0, aligned, resolved) == (100, 152, 1)

    def test_max_divergences_bounds_the_loop(self):
        sched = _make_scheduler()
        new, _, aligned, resolved = sched._splice_against_stored(
            PROMPT_M, STORED_M, max_divergences=1
        )
        assert new == A + [10, 11] + B + [20, 21] + D + C
        assert (aligned, resolved) == (152, 1)

    def test_many_divergences_within_default_bound(self):
        sched = _make_scheduler()
        new, p0, aligned, resolved = sched._splice_against_stored(
            PROMPT_MANY, STORED_MANY
        )
        assert new == STORED_MANY + C
        assert (p0, aligned, resolved) == (100, len(STORED_MANY), 70)

    def test_replacement_char_never_matches(self):
        sched = _make_scheduler()
        stored = A + [30] + B
        prompt = A + [31] + B + C  # different invalid bytes, both decode to U+FFFD
        new, _, _, resolved = sched._splice_against_stored(prompt, stored)
        assert new is prompt and resolved == 0

    def test_context_widened_past_partial_character(self):
        # The 4-token left context starts inside "€" (bytes 71, 72): the
        # decode is only clean once the context widens to include byte 70.
        sched = _make_scheduler()
        euro = [70, 71, 72, 73, 74]
        stored = A + euro + [10, 11] + B
        prompt = A + euro + [12] + B + C
        new, p0, aligned, resolved = sched._splice_against_stored(prompt, stored)
        assert new == A + euro + [10, 11] + B + C
        assert (p0, aligned, resolved) == (105, len(stored), 1)

    def test_special_token_never_spliced(self):
        sched = _make_scheduler()
        stored = A + [10, 99] + B  # "ab" + special id spelling "c"
        new, _, _, resolved = sched._splice_against_stored(PROMPT, stored)
        assert new is PROMPT and resolved == 0

    def test_stored_sequence_ending_inside_window(self):
        sched = _make_scheduler()
        stored = A + [10, 11]  # stored ends right after the sampled split
        new, p0, aligned, resolved = sched._splice_against_stored(PROMPT, stored)
        assert new == SPLICED
        assert (p0, aligned, resolved) == (100, 102, 1)

    def test_tail_of_prompt_is_never_rewritten(self):
        sched = _make_scheduler()
        prompt = A + [12] + B[:10]  # divergence inside the last 16 tokens
        new, _, _, resolved = sched._splice_against_stored(prompt, STORED)
        assert new is prompt and resolved == 0

    def test_decode_failure_is_noop(self):
        sched = _make_scheduler()
        sched.tokenizer = MagicMock()
        sched.tokenizer.decode.side_effect = RuntimeError("boom")
        new, _, _, resolved = sched._splice_against_stored(PROMPT, STORED)
        assert new is PROMPT and resolved == 0

    def test_long_prefix_with_array_sequence(self):
        sched = _make_scheduler()
        long_a = list(range(10000, 15000))  # exercises the chunked compare
        stored = array("i", long_a + [10, 11] + B)
        prompt = long_a + [12] + B + C
        new, p0, aligned, resolved = sched._splice_against_stored(prompt, stored)
        assert new == long_a + [10, 11] + B + C
        assert (p0, aligned, resolved) == (5000, len(stored), 1)


def _request(prompt_ids, request_id="req-new", **fields):
    req = Request(
        request_id=request_id,
        prompt=list(prompt_ids),
        sampling_params=SamplingParams(),
    )
    req.prompt_token_ids = list(prompt_ids)
    req.num_prompt_tokens = len(prompt_ids)
    for name, value in fields.items():
        setattr(req, name, value)
    return req


def _assert_untouched(req, prompt_ids):
    assert req.prompt_token_ids == prompt_ids
    assert req.num_prompt_tokens == len(prompt_ids)


def _infos(caplog):
    return [r for r in caplog.records if r.levelno == logging.INFO]


class TestMaybeSplicePrompt:
    def test_splices_request_and_logs_one_info_line(self, caplog):
        sched = _make_scheduler(block_size=64)
        sched._cache_probe_seqs.append(("req-old", array("i", STORED)))
        req = _request(PROMPT)

        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._maybe_splice_prompt_to_stored(req)

        assert req.prompt_token_ids == SPLICED
        assert req.num_prompt_tokens == len(SPLICED)
        assert Scheduler._common_prefix_len(req.prompt_token_ids, STORED) == len(STORED)
        infos = _infos(caplog)
        assert len(infos) == 1
        msg = infos[0].getMessage()
        assert (
            "prefix splice: request req-new re-aligned 1 sampled-token "
            "divergence(s) with stored req-old" in msg
        )
        # floor(152/64)*64 - floor(100/64)*64 = 128 - 64
        assert "matchable prefix 100 -> 152 of 152 stored tokens (+64 reusable)" in msg
        assert "prompt 181 -> 182 tokens" in msg
        assert "abc" not in msg  # no decoded text at INFO

    def test_second_call_is_noop(self, caplog):
        sched = _make_scheduler(block_size=64)
        sched._cache_probe_seqs.append(("req-old", array("i", STORED)))
        req = _request(PROMPT)
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._maybe_splice_prompt_to_stored(req)
            sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == SPLICED
        assert len(_infos(caplog)) == 1

    def test_picks_best_stored_sequence(self, caplog):
        sched = _make_scheduler(block_size=64)
        sched._cache_probe_seqs.append(("req-a", array("i", [7] * 50)))
        sched._cache_probe_seqs.append(("req-b", array("i", STORED)))
        sched._cache_probe_seqs.append(("req-c", array("i", A[:60] + [8] * 100)))
        req = _request(PROMPT)
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == SPLICED
        assert "with stored req-b" in caplog.text

    def test_newest_conversation_entry_is_used(self, caplog):
        # Turn 3: both stored turns tie at p0 = 100 (the turn-1 split), but
        # only the turn-2 entry also covers the turn-2 output.
        e = list(range(6000, 6030))
        s1 = A + [10, 11] + B
        s2 = s1 + C + [22, 23] + D
        prompt = A + [12] + B + C + [20, 21] + D + e
        sched = _make_scheduler(block_size=64)
        sched._cache_probe_seqs.append(("req-1", array("i", s1)))
        sched._cache_probe_seqs.append(("req-2", array("i", s2)))
        req = _request(prompt)
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == s2 + e
        msg = _infos(caplog)[0].getMessage()
        assert "with stored req-2" in msg
        assert f"matchable prefix 100 -> {len(s2)} of {len(s2)} stored tokens" in msg

    def test_ties_pick_newest(self, caplog):
        sched = _make_scheduler(block_size=64)
        sched._cache_probe_seqs.append(("req-x", array("i", STORED)))
        sched._cache_probe_seqs.append(("req-y", array("i", STORED)))
        req = _request(PROMPT)
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == SPLICED
        assert "with stored req-y" in caplog.text

    def test_many_historical_divergences_still_splice(self):
        # 70 re-aligned splits, cumulative growth 70: no fixed cap may stop it.
        sched = _make_scheduler(block_size=64)
        sched._cache_probe_seqs.append(("req-old", array("i", STORED_MANY)))
        req = _request(PROMPT_MANY)
        sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == STORED_MANY + C
        assert req.num_prompt_tokens == len(PROMPT_MANY) + 70

    def test_no_block_gained_is_noop(self, caplog):
        # Aligned 152 of 193 stored tokens: floor(152/2048) == floor(100/2048).
        sched = _make_scheduler(block_size=2048)
        sched._cache_probe_seqs.append(("req-old", array("i", STORED_2)))
        req = _request(PROMPT_2)
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT_2)
        assert not _infos(caplog)

    def test_trailing_partial_block_is_not_counted(self):
        # Stored ends at 112 (aligned 112): floor(112/64) == floor(100/64).
        sched = _make_scheduler(block_size=64)
        sched._cache_probe_seqs.append(("req-old", array("i", A + [10, 11] + B[:10])))
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)

    def test_partial_alignment_with_block_gain_splices(self):
        sched = _make_scheduler(block_size=64)
        sched._cache_probe_seqs.append(("req-old", array("i", STORED_2)))
        req = _request(PROMPT_2)
        sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == A + [10, 11] + B + [20] + D + C

    def test_exact_hit_is_never_created(self):
        # 182 = 2 * 91: the fetch would cover the whole spliced prompt, and an
        # exact hit makes stateful caches re-prefill, so it is not applied.
        sched = _make_scheduler(block_size=91)
        sched._cache_probe_seqs.append(("req-old", array("i", SPLICED)))
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)

    def test_prompt_that_is_a_prefix_of_a_stored_entry_still_splices(self):
        # Retry of a stored prompt: the fetch covers 128 of 182 tokens, so a
        # suffix stays to prefill and the splice is applied.
        sched = _make_scheduler(block_size=64)
        sched._cache_probe_seqs.append(("req-old", array("i", SPLICED)))
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == SPLICED

    def test_probe_keeps_eight_entries(self):
        assert getattr(Scheduler, "_CACHE_PROBE_MAXLEN", None) == 8

    def test_env_opt_out(self, monkeypatch):
        monkeypatch.setenv("OMLX_DISABLE_SAMPLED_SPLICE", "1")
        sched = _make_scheduler(block_size=64)
        sched._cache_probe_seqs.append(("req-old", array("i", STORED)))
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("skip_cache_store", True),
            ("benchmark_trace", True),
            ("images", ["img"]),
            ("videos", ["clip"]),
            ("vlm_inputs_embeds", object()),
            ("vlm_extra_kwargs", {"position_ids": 1}),
            ("vlm_image_hash", "img-hash"),
            ("vlm_cache_key_start", 5),
            ("vlm_cache_key_ranges", [(0, "img-hash")]),
            ("specprefill_indices", object()),
            ("specprefill_system_end", 120),
        ],
    )
    def test_guarded_request_fields_are_noop(self, field, value):
        sched = _make_scheduler(block_size=64)
        sched._cache_probe_seqs.append(("req-old", array("i", STORED)))
        req = _request(PROMPT, **{field: value})
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)

    def test_no_block_aware_cache_is_noop(self):
        sched = _make_scheduler(block_size=64)
        sched.block_aware_cache = None
        sched._cache_probe_seqs.append(("req-old", array("i", STORED)))
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)

    def test_unreconstructible_cache_is_noop(self):
        sched = _make_scheduler(block_size=64)
        sched._unreconstructible_cache_model = True
        sched._cache_probe_seqs.append(("req-old", array("i", STORED)))
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)

    def test_tokenizer_without_decode_is_noop(self):
        sched = _make_scheduler(block_size=64)
        sched.tokenizer = SimpleNamespace(encode=lambda text, **kw: [])
        sched._cache_probe_seqs.append(("req-old", array("i", STORED)))
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)

    def test_missing_or_empty_probe_is_noop(self):
        sched = _make_scheduler(block_size=64)
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)  # empty deque
        _assert_untouched(req, PROMPT)
        del sched._cache_probe_seqs  # scheduler built without the probe
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)


GEN = [900, 901]  # generation prompt "<900><901>" encoded from "<GEN>"


def _make_admission_scheduler(encode_map=None):
    sched = _make_scheduler(block_size=64, encode_map=encode_map)
    sched.requests = {}
    sched.waiting = deque()
    sched.moe_offload_stats = None
    sched._detect_boundary_snapshot_need = lambda: False
    return sched


class TestAddRequestSplice:
    def test_list_prompt_spliced_before_generation_prompt_is_resolved(self):
        sched = _make_admission_scheduler(encode_map={"<GEN>": GEN})
        sched._cache_probe_seqs.append(("req-old", array("i", STORED)))
        req = Request(
            request_id="req-new",
            prompt=PROMPT + GEN,
            sampling_params=SamplingParams(),
            generation_prompt_text="<GEN>",
        )

        sched.add_request(req)

        expected = SPLICED + GEN
        assert req.prompt_token_ids == expected
        assert req.num_prompt_tokens == len(expected)
        assert req.generation_prompt_start == len(expected) - len(GEN)
        assert Scheduler._common_prefix_len(req.prompt_token_ids, STORED) == len(STORED)
        assert sched.waiting[-1] is req

    def test_string_prompt_spliced(self):
        sched = _make_admission_scheduler(encode_map={"TEXT": PROMPT})
        sched._cache_probe_seqs.append(("req-old", array("i", STORED)))
        req = Request(
            request_id="req-new", prompt="TEXT", sampling_params=SamplingParams()
        )

        sched.add_request(req)

        assert req.prompt_token_ids == SPLICED
        assert req.num_prompt_tokens == len(SPLICED)

    def test_only_requests_tokenized_here_are_spliced(self):
        sched = _make_admission_scheduler()
        sched._cache_probe_seqs.append(("req-old", array("i", STORED)))
        fresh = Request(
            request_id="req-fresh",
            prompt=list(PROMPT),
            sampling_params=SamplingParams(),
        )
        pre = Request(
            request_id="req-pre", prompt="ignored", sampling_params=SamplingParams()
        )
        pre.prompt_token_ids = list(PROMPT)
        pre.num_prompt_tokens = len(PROMPT)

        sched.add_request(fresh)
        sched.add_request(pre)

        assert fresh.prompt_token_ids == SPLICED
        assert pre.prompt_token_ids == PROMPT
        assert pre.num_prompt_tokens == len(PROMPT)

# SPDX-License-Identifier: Apache-2.0
"""Tests for the prompt-side splice of re-sent sampled output tokens."""

import concurrent.futures
import hashlib
import logging
import threading
from array import array
from collections import OrderedDict, deque
from contextlib import nullcontext
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


def _make_scheduler(block_size=64, encode_map=None, ssd_dir=None):
    sched = object.__new__(Scheduler)
    sched._cache_probe_seqs = deque(maxlen=8)
    sched._splice_refs = OrderedDict()
    sched._splice_refs_lock = threading.Lock()
    sched.tokenizer = _PieceTokenizer(encode_map=encode_map)
    sched.config = MagicMock(spec=[])
    sched.config.paged_cache_block_size = block_size
    sched.config.max_num_seqs = 4
    sched.config.paged_ssd_cache_dir = None if ssd_dir is None else str(ssd_dir)
    sched.config.hot_cache_only = False
    sched.config.model_name = "test-model"
    sched.block_aware_cache = object()
    sched._unreconstructible_cache_model = False
    return sched


def _ref(sched, request_id, seq):
    """Register a stored sequence as the store submit path does."""
    sched._register_splice_ref(request_id, array("i", seq))


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
        _ref(sched, "req-old", STORED)
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
        _ref(sched, "req-old", STORED)
        req = _request(PROMPT)
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._maybe_splice_prompt_to_stored(req)
            sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == SPLICED
        assert len(_infos(caplog)) == 1

    def test_picks_best_stored_sequence(self, caplog):
        sched = _make_scheduler(block_size=64)
        _ref(sched, "req-a", [7] * 50)
        _ref(sched, "req-b", STORED)
        _ref(sched, "req-c", A[:60] + [8] * 100)
        req = _request(PROMPT)
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == SPLICED
        assert "with stored req-b" in caplog.text

    def test_newest_conversation_entry_is_used(self, caplog):
        # Turn 3: both stored turns tie at p0 = 100 (the turn-1 split), but
        # only the turn-2 entry also covers the turn-2 output. Registering
        # req-2 would supersede req-1, so both are placed directly (as two
        # references sharing a prefix can be, e.g. a regenerated turn).
        e = list(range(6000, 6030))
        s1 = A + [10, 11] + B
        s2 = s1 + C + [22, 23] + D
        prompt = A + [12] + B + C + [20, 21] + D + e
        sched = _make_scheduler(block_size=64)
        sched._splice_refs["req-1"] = array("i", s1)
        sched._splice_refs["req-2"] = array("i", s2)
        req = _request(prompt)
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == s2 + e
        msg = _infos(caplog)[0].getMessage()
        assert "with stored req-2" in msg
        assert f"matchable prefix 100 -> {len(s2)} of {len(s2)} stored tokens" in msg

    def test_ties_pick_newest(self, caplog):
        sched = _make_scheduler(block_size=64)
        # Placed directly: registering req-y would supersede req-x.
        sched._splice_refs["req-x"] = array("i", STORED)
        sched._splice_refs["req-y"] = array("i", STORED)
        req = _request(PROMPT)
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == SPLICED
        assert "with stored req-y" in caplog.text

    def test_many_historical_divergences_still_splice(self):
        # 70 re-aligned splits, cumulative growth 70: no fixed cap may stop it.
        sched = _make_scheduler(block_size=64)
        _ref(sched, "req-old", STORED_MANY)
        req = _request(PROMPT_MANY)
        sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == STORED_MANY + C
        assert req.num_prompt_tokens == len(PROMPT_MANY) + 70

    def test_no_block_gained_is_noop(self, caplog):
        # Aligned 152 of 193 stored tokens: floor(152/2048) == floor(100/2048).
        sched = _make_scheduler(block_size=2048)
        _ref(sched, "req-old", STORED_2)
        req = _request(PROMPT_2)
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT_2)
        assert not _infos(caplog)

    def test_trailing_partial_block_is_not_counted(self):
        # Stored ends at 112 (aligned 112): floor(112/64) == floor(100/64).
        sched = _make_scheduler(block_size=64)
        _ref(sched, "req-old", A + [10, 11] + B[:10])
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)

    def test_partial_alignment_with_block_gain_splices(self):
        sched = _make_scheduler(block_size=64)
        _ref(sched, "req-old", STORED_2)
        req = _request(PROMPT_2)
        sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == A + [10, 11] + B + [20] + D + C

    def test_exact_hit_is_never_created(self):
        # 182 = 2 * 91: the fetch would cover the whole spliced prompt, and an
        # exact hit makes stateful caches re-prefill, so it is not applied.
        sched = _make_scheduler(block_size=91)
        _ref(sched, "req-old", SPLICED)
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)

    def test_prompt_that_is_a_prefix_of_a_stored_entry_still_splices(self):
        # Retry of a stored prompt: the fetch covers 128 of 182 tokens, so a
        # suffix stays to prefill and the splice is applied.
        sched = _make_scheduler(block_size=64)
        _ref(sched, "req-old", SPLICED)
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == SPLICED

    def test_probe_keeps_eight_entries(self):
        assert getattr(Scheduler, "_CACHE_PROBE_MAXLEN", None) == 8

    def test_env_opt_out(self, monkeypatch):
        monkeypatch.setenv("OMLX_DISABLE_SAMPLED_SPLICE", "1")
        sched = _make_scheduler(block_size=64)
        _ref(sched, "req-old", STORED)
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
        _ref(sched, "req-old", STORED)
        req = _request(PROMPT, **{field: value})
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)

    def test_no_block_aware_cache_is_noop(self):
        sched = _make_scheduler(block_size=64)
        sched.block_aware_cache = None
        _ref(sched, "req-old", STORED)
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)

    def test_unreconstructible_cache_is_noop(self):
        sched = _make_scheduler(block_size=64)
        sched._unreconstructible_cache_model = True
        _ref(sched, "req-old", STORED)
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)

    def test_tokenizer_without_decode_is_noop(self):
        sched = _make_scheduler(block_size=64)
        sched.tokenizer = SimpleNamespace(encode=lambda text, **kw: [])
        _ref(sched, "req-old", STORED)
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)

    def test_missing_or_empty_reference_store_is_noop(self):
        sched = _make_scheduler(block_size=64)
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)  # empty deque
        _assert_untouched(req, PROMPT)
        del sched._splice_refs  # scheduler built without the reference store
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
        _ref(sched, "req-old", STORED)
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
        _ref(sched, "req-old", STORED)
        req = Request(
            request_id="req-new", prompt="TEXT", sampling_params=SamplingParams()
        )

        sched.add_request(req)

        assert req.prompt_token_ids == SPLICED
        assert req.num_prompt_tokens == len(SPLICED)

    def test_only_requests_tokenized_here_are_spliced(self):
        sched = _make_admission_scheduler()
        _ref(sched, "req-old", STORED)
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


def _ref_dir(ssd_dir, model_name="test-model"):
    model_key = hashlib.sha256(model_name.encode()).hexdigest()[:16]
    return ssd_dir / "splice_refs" / model_key


def _ref_file(ssd_dir, request_id, model_name="test-model"):
    name = hashlib.sha256(request_id.encode()).hexdigest()[:32] + ".i32"
    return _ref_dir(ssd_dir, model_name) / name


def _unrelated(k):
    return list(range(7000 + 100 * k, 7050 + 100 * k))


def _make_drain_scheduler(**kwargs):
    sched = _make_scheduler(**kwargs)
    sched._pending_async_removes = deque()
    sched.uid_to_request_id = {}
    sched.request_id_to_uid = {}
    sched._inflight_store_futures = {}
    sched._inflight_store_info = {}
    sched._boundary_snapshot_store = None
    sched._store_cache_gate = None
    sched.requests = {}
    sched._clear_request_admission_bookkeeping = lambda request_id: None
    sched._schedule_deferred_metal_clear = lambda: None
    return sched


def _done_future(result=None, exc=None):
    future = concurrent.futures.Future()
    if exc is not None:
        future.set_exception(exc)
    else:
        future.set_result(result)
    return future


class TestSpliceReferenceStore:
    def test_unrelated_completions_do_not_evict_conversation_ref(self, caplog):
        sched = _make_scheduler(block_size=64)
        _ref(sched, "req-conv", STORED)
        for k in range(20):
            _ref(sched, f"req-side-{k}", _unrelated(k))
        req = _request(PROMPT)
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == SPLICED
        assert "with stored req-conv" in caplog.text

    def test_newer_turn_supersedes_older_ref(self, tmp_path):
        s1 = A + [10, 11] + B
        s2 = s1 + C + [22, 23] + D
        sched = _make_scheduler(block_size=64, ssd_dir=tmp_path)
        _ref(sched, "req-1", s1)
        sched._persist_splice_ref("req-1")
        assert _ref_file(tmp_path, "req-1").exists()
        _ref(sched, "req-side", _unrelated(0))
        _ref(sched, "req-2", s2)
        assert list(sched._splice_refs) == ["req-side", "req-2"]
        assert list(sched._splice_refs["req-2"]) == s2
        assert not _ref_file(tmp_path, "req-1").exists()

    def test_diverging_ref_does_not_supersede(self):
        sched = _make_scheduler(block_size=64)
        _ref(sched, "req-1", STORED)
        _ref(sched, "req-2", A + [15] + B)  # same prefix, then diverges
        assert list(sched._splice_refs) == ["req-1", "req-2"]

    def test_using_a_ref_moves_it_to_most_recent(self):
        sched = _make_scheduler(block_size=64)
        sched._SPLICE_REF_MAX_ENTRIES = 3
        _ref(sched, "req-conv", STORED)
        _ref(sched, "req-side-0", _unrelated(0))
        _ref(sched, "req-side-1", _unrelated(1))
        sched._maybe_splice_prompt_to_stored(_request(PROMPT))
        _ref(sched, "req-side-2", _unrelated(2))  # evicts req-side-0, not conv
        assert list(sched._splice_refs) == ["req-side-1", "req-conv", "req-side-2"]

    def test_entry_cap_evicts_least_recent(self, tmp_path):
        sched = _make_scheduler(block_size=64, ssd_dir=tmp_path)
        sched._SPLICE_REF_MAX_ENTRIES = 3
        for k in range(4):
            _ref(sched, f"req-{k}", _unrelated(k))
            sched._persist_splice_ref(f"req-{k}")
        assert list(sched._splice_refs) == ["req-1", "req-2", "req-3"]
        assert not _ref_file(tmp_path, "req-0").exists()
        assert _ref_file(tmp_path, "req-3").exists()

    def test_token_cap_evicts_least_recent(self):
        sched = _make_scheduler(block_size=64)
        sched._SPLICE_REF_MAX_TOKENS = 120
        _ref(sched, "req-0", _unrelated(0))  # 50 tokens
        _ref(sched, "req-1", _unrelated(1))  # 100
        _ref(sched, "req-2", _unrelated(2))  # 150 > 120: req-0 goes
        assert list(sched._splice_refs) == ["req-1", "req-2"]

    def test_failed_store_drops_ref(self, tmp_path):
        sched = _make_drain_scheduler(block_size=64, ssd_dir=tmp_path)
        _ref(sched, "req-old", STORED)
        sched._persist_splice_ref("req-old")
        sched._pending_async_removes.append((0, "req-old", _done_future(False)))
        sched._drain_pending_async_removes()
        assert "req-old" not in sched._splice_refs
        assert not _ref_file(tmp_path, "req-old").exists()
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)

    def test_raised_store_drops_ref(self):
        sched = _make_drain_scheduler(block_size=64)
        _ref(sched, "req-old", STORED)
        future = _done_future(exc=RuntimeError("boom"))
        sched._pending_async_removes.append((0, "req-old", future))
        sched._drain_pending_async_removes()
        assert "req-old" not in sched._splice_refs

    def test_successful_store_keeps_ref(self):
        sched = _make_drain_scheduler(block_size=64)
        _ref(sched, "req-old", STORED)
        sched._pending_async_removes.append((0, "req-old", _done_future(True)))
        sched._drain_pending_async_removes()
        assert "req-old" in sched._splice_refs

    def test_worker_reports_failure_and_does_not_persist(self, tmp_path):
        sched = _make_scheduler(block_size=64, ssd_dir=tmp_path)
        sched._stream = None
        sched._phase_timer = lambda name: nullcontext()
        sched.paged_cache_manager = None
        sched.block_aware_cache = MagicMock()
        sched.block_aware_cache.store_cache.side_effect = RuntimeError("disk")
        _ref(sched, "req-old", STORED)
        ok = sched._async_store_cache_worker(
            "req-old", list(STORED), [], None, None, None, None, None
        )
        assert ok is False
        assert not _ref_file(tmp_path, "req-old").exists()
        sched.block_aware_cache.store_cache.side_effect = None
        sched.block_aware_cache.store_cache.return_value = None
        ok = sched._async_store_cache_worker(
            "req-old", list(STORED), [], None, None, None, None, None
        )
        assert ok is True
        assert _ref_file(tmp_path, "req-old").exists()

    def test_refs_survive_restart(self, tmp_path, caplog):
        # Turn 2 was spliced and stored; the server restarts before turn 3.
        e = list(range(6000, 6030))
        s2 = A + [10, 11] + B + C + [22, 23] + D
        before = _make_scheduler(block_size=64, ssd_dir=tmp_path)
        _ref(before, "req-2", s2)
        before._persist_splice_ref("req-2")

        after = _make_scheduler(block_size=64, ssd_dir=tmp_path)
        assert not after._splice_refs
        prompt = A + [12] + B + C + [20, 21] + D + e
        req = _request(prompt)
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            after._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == s2 + e
        assert f"matchable prefix 100 -> {len(s2)}" in caplog.text

    def test_restart_load_applies_supersede_and_caps(self, tmp_path):
        before = _make_scheduler(block_size=64, ssd_dir=tmp_path)
        for k in range(3):
            _ref(before, f"req-{k}", _unrelated(k))
            before._persist_splice_ref(f"req-{k}")
        after = _make_scheduler(block_size=64, ssd_dir=tmp_path)
        after._SPLICE_REF_MAX_ENTRIES = 2
        _ref(after, "req-new", _unrelated(0) + [5])  # supersedes loaded req-0
        assert len(after._splice_refs) == 2
        assert "req-new" in after._splice_refs
        assert not _ref_file(tmp_path, "req-0").exists()
        assert not _ref_file(tmp_path, "req-1").exists()
        assert _ref_file(tmp_path, "req-2").exists()

    def test_other_models_refs_are_not_loaded(self, tmp_path):
        before = _make_scheduler(block_size=64, ssd_dir=tmp_path)
        _ref(before, "req-old", STORED)
        before._persist_splice_ref("req-old")
        other = _make_scheduler(block_size=64, ssd_dir=tmp_path)
        other.config.model_name = "other-model"
        req = _request(PROMPT)
        other._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)
        assert _ref_file(tmp_path, "req-old").exists()

    def test_hot_cache_only_does_not_persist(self, tmp_path):
        sched = _make_scheduler(block_size=64, ssd_dir=tmp_path)
        sched.config.hot_cache_only = True
        _ref(sched, "req-old", STORED)
        sched._persist_splice_ref("req-old")
        assert not (tmp_path / "splice_refs").exists()

    def test_persistence_errors_are_ignored(self, tmp_path):
        not_a_dir = tmp_path / "file"
        not_a_dir.write_bytes(b"x")
        sched = _make_scheduler(block_size=64, ssd_dir=not_a_dir)
        _ref(sched, "req-old", STORED)
        sched._persist_splice_ref("req-old")  # mkdir fails: no raise
        sched._drop_splice_ref("req-old")
        _ref(sched, "req-old", STORED)
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        assert req.prompt_token_ids == SPLICED  # in-memory ref still works

    def test_garbage_files_are_ignored(self, tmp_path):
        ref_dir = _ref_dir(tmp_path)
        ref_dir.mkdir(parents=True)
        odd = ref_dir / ("0" * 32 + ".i32")
        odd.write_bytes(b"\x01\x02\x03")
        junk = ref_dir / ("1" * 32 + ".i32")
        junk.write_bytes(array("i", [123456] * 200).tobytes())
        (ref_dir / ("2" * 32 + ".i32")).mkdir()  # unreadable entry
        sched = _make_scheduler(block_size=64, ssd_dir=tmp_path)
        req = _request(PROMPT)
        sched._maybe_splice_prompt_to_stored(req)
        _assert_untouched(req, PROMPT)
        assert not odd.exists()

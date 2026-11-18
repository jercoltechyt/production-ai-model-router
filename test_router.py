# -*- coding: utf-8 -*-
"""
Unit test suite for the Production 3-Tier AI Model Router.
Run with: python test_router.py (or pytest -q)
"""

from __future__ import annotations

import asyncio
import unittest
from cost_simulator import simulate_workload_mix
from providers import (
    build_anthropic_payload,
    build_gemini_payload,
    build_openai_payload,
    create_live_or_mock_invoker,
)
from router import (
    PrefixCacheTracker,
    TIERS,
    estimate_call_cost_usd,
    route_with_cascade,
    route_with_cascade_async,
    select_initial_tier,
    select_tier,
)


class TestProductionRouter(unittest.TestCase):
    def test_video_signature_parity_select_tier(self) -> None:
        tier, ms = select_tier("Reset user password link", cached_tokens=0, strict_schema=False)
        self.assertEqual(tier, "fast")
        self.assertLess(ms, 5.0)
        # Verify alias and dict-style TierConfig access shown in the video
        tier_alias, _ = select_initial_tier("Reset user password link", cached_prefix_tokens=0)
        self.assertEqual(tier_alias, "fast")
        self.assertEqual(TIERS["fast"]["model"], "gemini-3.8-flash-lite")
        self.assertEqual(TIERS["balanced"]["effort"], "medium")

    def test_cache_pinning_protects_hot_prefix(self) -> None:
        tier, _ = select_tier("Short follow-up question", cached_tokens=16384, strict_schema=False)
        self.assertEqual(tier, "balanced")

    def test_prefix_cache_tracker_ttl_and_sha256(self) -> None:
        tracker = PrefixCacheTracker(ttl_seconds=300.0, min_cacheable_tokens=4096)
        long_rag_prefix = "legal_clause " * 4500  # >4,096 words/tokens
        self.assertEqual(tracker.get_warm_tokens(long_rag_prefix, now=100.0), 0)

        recorded = tracker.record_prefix(long_rag_prefix, tier="balanced", now=100.0)
        self.assertGreaterEqual(recorded, 4096)
        self.assertEqual(tracker.get_warm_tokens(long_rag_prefix, now=250.0), recorded)
        # After TTL expiration (100 + 300 = 400s), returns 0
        self.assertEqual(tracker.get_warm_tokens(long_rag_prefix, now=401.0), 0)

    def test_deep_tier_for_architecture_and_formal_proofs(self) -> None:
        tier, _ = select_tier(
            "Prove deadlock freedom and optimize lock-free ring buffer architecture",
            cached_tokens=0,
            strict_schema=False,
        )
        self.assertEqual(tier, "deep")

    def test_cascade_fallback_rescues_truncated_json(self) -> None:
        def flaky_invoker(model: str, effort: str) -> str:
            if "flash-lite" in model:
                return '{"invoice_id": "INV-100", '  # Truncated JSON on fast tier
            return '{"invoice_id": "INV-100", "status": "paid"}'

        result = route_with_cascade(
            prompt="Parse webhook JSON",
            invoker=flaky_invoker,
            required_keys={"invoice_id", "status"},
            cached_tokens=0,
        )
        self.assertEqual(result["initial_tier"], "fast")
        self.assertEqual(result["tier"], "balanced")
        self.assertEqual(result["final_tier"], "balanced")
        self.assertTrue(result["escalated"])
        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["attempts"], ["fast", "balanced"])
        self.assertEqual(result["data"]["status"], "paid")

    def test_markdown_fence_stripping_and_typed_schema_validation(self) -> None:
        def fenced_bad_type_on_fast(model: str, effort: str) -> str:
            if "flash-lite" in model:
                # Valid JSON inside fences, but total_usd is a string instead of float/int
                return '```json\n{"invoice_id": "INV-200", "total_usd": "not_a_number"}\n```'
            return '```json\n{"invoice_id": "INV-200", "total_usd": 1250.75}\n```'

        res = route_with_cascade(
            prompt="Extract typed invoice",
            invoker=fenced_bad_type_on_fast,
            required_keys={"invoice_id": str, "total_usd": (int, float)},
            cached_tokens=0,
        )
        self.assertTrue(res["escalated"])
        self.assertEqual(res["tier"], "balanced")
        self.assertEqual(res["data"]["total_usd"], 1250.75)

    def test_async_route_with_cascade(self) -> None:
        async def async_flaky(model: str, effort: str) -> str:
            await asyncio.sleep(0.001)
            if "flash-lite" in model:
                return '{"broken": '
            return '{"invoice_id": "INV-ASYNC", "status": "ok"}'

        res = asyncio.run(
            route_with_cascade_async(
                prompt="Async webhook parse",
                async_invoker=async_flaky,
                required_keys={"invoice_id", "status"},
                cached_tokens=0,
            )
        )
        self.assertTrue(res["escalated"])
        self.assertEqual(res["tier"], "balanced")
        self.assertEqual(res["data"]["invoice_id"], "INV-ASYNC")

    def test_cost_calculation_respects_cached_discount(self) -> None:
        cost = estimate_call_cost_usd("balanced", uncached_input_tokens=500, cached_input_tokens=20000, output_tokens=200)
        self.assertAlmostEqual(cost, 0.005, places=6)

    def test_cost_simulator_savings(self) -> None:
        sim = simulate_workload_mix(monthly_requests=100_000_000, avg_cached_tokens=24_000, fast_json_fail_rate=0.06)
        self.assertGreater(sim["savings_pct"], 75.0)
        self.assertLess(sim["warm_balanced_rag_call_usd"], sim["cold_fast_rag_call_usd"])

    def test_provider_payloads_and_dry_run_invoker(self) -> None:
        oai = build_openai_payload("gpt-6-astra", "Test", reasoning_effort="high", prompt_cache_key="tenant-42")
        self.assertIn("max_completion_tokens", oai)
        self.assertNotIn("max_tokens", oai)
        self.assertNotIn("temperature", oai)
        self.assertEqual(oai["reasoning"]["effort"], "high")
        self.assertEqual(oai["prompt_cache_key"], "tenant-42")

        ant = build_anthropic_payload("claude-sonnet-5.5", "Test", static_system_prefix="System context")
        self.assertEqual(ant["effort"], "medium")
        self.assertEqual(ant["system"][0]["cache_control"]["type"], "ephemeral")

        gem = build_gemini_payload("gemini-4-argon", "Test", reasoning_effort="low")
        self.assertEqual(gem["generationConfig"]["thinkingConfig"]["thinkingLevel"], "LOW")

        inv = create_live_or_mock_invoker("Test", static_system_prefix="ctx", dry_run=True)
        res = route_with_cascade("Test", inv, required_keys={"status"}, cached_tokens=8192)
        self.assertTrue(res["cache_hit"])
        self.assertEqual(res["tier"], "balanced")


if __name__ == "__main__":
    unittest.main(verbosity=2)

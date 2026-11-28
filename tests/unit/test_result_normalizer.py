"""Mandatory Result Comparator & Normalization Test Matrix (Blueprint Section 19.3)."""

from decimal import Decimal
import math
from sql_grpo.env.result_normalizer import (
    canonical_multiset_digest,
    compare_results,
)


def test_unordered_equivalence_and_duplicate_preservation():
    gold = [(1, "Alice"), (1, "Alice"), (2, "Bob")]
    reordered = [(2, "Bob"), (1, "Alice"), (1, "Alice")]
    missing_duplicate = [(1, "Alice"), (2, "Bob")]
    extra_duplicate = [(1, "Alice"), (2, "Bob"), (2, "Bob")]

    assert compare_results(reordered, gold, ordered=False) is True
    assert compare_results(missing_duplicate, gold, ordered=False) is False
    assert compare_results(extra_duplicate, gold, ordered=False) is False


def test_ordered_sequence_semantics():
    gold = [(1, "Alice"), (2, "Bob")]
    same_order = [(1, "Alice"), (2, "Bob")]
    swapped_order = [(2, "Bob"), (1, "Alice")]

    assert compare_results(same_order, gold, ordered=True) is True
    assert compare_results(swapped_order, gold, ordered=True) is False


def test_type_distinction_null_int_float_bool_text_decimal_blob():
    # NULL vs string "None" vs empty string
    assert compare_results([(None,)], [(None,)]) is True
    assert compare_results([(None,)], [("None",)]) is False
    assert compare_results([(None,)], [("",)]) is False

    # Integer vs Text vs Float vs Bool (strict distinction by default)
    assert compare_results([(1,)], [("1",)]) is False
    assert compare_results([(1,)], [(1.0,)]) is False
    assert compare_results([(True,)], [(1,)]) is False
    assert compare_results([(1,)], [(1.0,)], coerce_numeric=True) is True

    # Decimal equivalence
    assert compare_results([(Decimal("1.500"),)], [(Decimal("1.5"),)]) is True
    assert compare_results([(Decimal("1.500"),)], [(Decimal("1.51"),)]) is False

    # Case & whitespace preserved strictly by default
    assert compare_results([("Alice",)], [("alice",)]) is False
    assert compare_results([("Alice",)], [(" Alice ",)]) is False
    assert compare_results([("Alice",)], [(" Alice ",)], strip_text=True) is True

    # Unicode text & BLOB bytes
    assert compare_results([("café 東京",)], [("café 東京",)]) is True
    assert compare_results([(b"\x00\xab",)], [(b"\x00\xab",)]) is True
    assert compare_results([(b"\x00\xab",)], [(b"\x00\xac",)]) is False


def test_float_tolerance_and_special_floats():
    # Exact floats by default (float_tolerance=0.0)
    assert compare_results([(1.00004,)], [(1.00006,)], float_tolerance=0.0) is False
    # Tolerance compares actual distance without fixed round(4) boundary errors
    assert compare_results([(1.00004,)], [(1.00006,)], float_tolerance=1e-4) is True
    assert (
        compare_results(
            [("a", 1.00004), ("b", 2.5)],
            [("b", 2.50002), ("a", 1.00006)],
            ordered=False,
            float_tolerance=1e-4,
        )
        is True
    )
    assert compare_results([(1.0,)], [(1.1,)], float_tolerance=1e-4) is False

    # Special IEEE-754 floats (NaN, -0.0, inf)
    assert compare_results([(float("nan"),)], [(float("nan"),)]) is True
    assert compare_results([(-0.0,)], [(0.0,)]) is True
    assert compare_results([(float("inf"),)], [(float("-inf"),)]) is False


def test_truncated_empty_and_large_multiset_digest():
    assert compare_results([(1,)], [(1,)], candidate_truncated=True) is False
    assert compare_results([], [(1,)]) is False
    assert compare_results([], []) is True

    # Deterministic multiset digest is order-invariant for unordered queries and duplicate-sensitive
    d1 = canonical_multiset_digest([(i, f"v{i % 7}") for i in range(1000)], ordered=False)
    d2 = canonical_multiset_digest([(i, f"v{i % 7}") for i in reversed(range(1000))], ordered=False)
    d3 = canonical_multiset_digest([(i, f"v{i % 7}") for i in reversed(range(1000))], ordered=True)
    assert d1 == d2
    assert d1 != d3

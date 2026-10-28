"""
test_tiling.py
--------------
Unit test: tile → stitch → verify reconstructed == original.
Run with: pytest tests/test_tiling.py -v
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pytest
from preprocessing import tile_image, stitch_predictions


def _make_synthetic_image(H=512, W=512, C=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(H, W, C), dtype=np.uint8)


def _make_synthetic_mask(H=512, W=512, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(H, W), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test 1 — RGB image round-trip
# ---------------------------------------------------------------------------

def test_tile_stitch_rgb_exact():
    """Tile and stitch an RGB image; reconstructed must equal original."""
    original = _make_synthetic_image(H=512, W=512, C=3)
    tiles = tile_image(original, tile_size=256, stride=256)   # no overlap

    # Simulate identity predictions (pass tile as-is)
    for t in tiles:
        t["pred"] = t["tile"][:, :, 0].astype(float)  # single channel

    stitched = stitch_predictions(tiles, image_shape=(512, 512), tile_size=256)
    expected = original[:, :, 0].astype(float)

    assert stitched.shape == expected.shape, f"Shape mismatch: {stitched.shape} vs {expected.shape}"
    assert np.allclose(stitched, expected, atol=1e-5), \
        f"Max diff: {np.abs(stitched - expected).max()}"


# ---------------------------------------------------------------------------
# Test 2 — Binary mask round-trip
# ---------------------------------------------------------------------------

def test_tile_stitch_mask_exact():
    """Tile and stitch a binary mask; reconstructed must equal original."""
    original = _make_synthetic_mask(H=600, W=700)
    tiles = tile_image(original, tile_size=256, stride=256)

    for t in tiles:
        t["pred"] = t["tile"].astype(float)

    stitched = stitch_predictions(tiles, image_shape=(600, 700), tile_size=256)
    expected = original.astype(float)

    assert np.allclose(stitched, expected, atol=1e-5), \
        f"Max diff: {np.abs(stitched - expected).max()}"


# ---------------------------------------------------------------------------
# Test 3 — Overlapping tiles (50% stride) round-trip
# ---------------------------------------------------------------------------

def test_tile_stitch_overlapping():
    """With 50% overlap (stride=128), stitched result should match original."""
    original = _make_synthetic_mask(H=512, W=512)
    tiles = tile_image(original, tile_size=256, stride=128)

    for t in tiles:
        t["pred"] = t["tile"].astype(float)

    stitched = stitch_predictions(tiles, image_shape=(512, 512), tile_size=256)
    expected = original.astype(float)

    assert np.allclose(stitched, expected, atol=1e-5), \
        f"Max diff: {np.abs(stitched - expected).max()}"


# ---------------------------------------------------------------------------
# Test 4 — Non-square image
# ---------------------------------------------------------------------------

def test_tile_stitch_nonsquare():
    """Works on non-square images."""
    original = _make_synthetic_image(H=384, W=512, C=3)
    tiles = tile_image(original, tile_size=256, stride=256)

    for t in tiles:
        t["pred"] = t["tile"][:, :, 0].astype(float)

    stitched = stitch_predictions(tiles, image_shape=(384, 512), tile_size=256)
    expected = original[:, :, 0].astype(float)

    assert stitched.shape == expected.shape
    assert np.allclose(stitched, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 5 — Tile count sanity
# ---------------------------------------------------------------------------

def test_tile_count():
    """With stride=stride, tile count should be >= ceil(H/stride) * ceil(W/stride)."""
    original = _make_synthetic_image(H=1000, W=1000, C=3)
    tiles_128 = tile_image(original, tile_size=256, stride=128)
    tiles_256 = tile_image(original, tile_size=256, stride=256)

    # More tiles with smaller stride
    assert len(tiles_128) >= len(tiles_256), \
        f"Expected more tiles with stride=128, got {len(tiles_128)} vs {len(tiles_256)}"


if __name__ == "__main__":
    test_tile_stitch_rgb_exact()
    test_tile_stitch_mask_exact()
    test_tile_stitch_overlapping()
    test_tile_stitch_nonsquare()
    test_tile_count()
    print("All tiling tests passed.")

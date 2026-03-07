from miniball.coordinate_transformations import (
    normalized_to_screen,
    screen_to_normalized,
)


def test_screen_to_normalized():
    # 0,0 maps to origin in both spaces
    assert screen_to_normalized(0, 0) == (0.0, 0.0)
    # 100 px on a 1200-wide screen → 100/1200 × 120 = 10 normalised units
    # 100 px on an 800-tall screen  → 100/800  × 80  = 10 normalised units
    assert screen_to_normalized(100, 100) == (10.0, 10.0)
    # 1000 px → 100.0 normalised units (both axes)
    assert screen_to_normalized(1000, 1000) == (100.0, 100.0)


def test_normalized_to_screen():
    assert normalized_to_screen(0, 0) == (0.0, 0.0)
    # 0.1 normalised → 0.1 × 1200/120 = 1.0 px  (X)
    # 0.1 normalised → 0.1 × 800/80   = 1.0 px  (Y)
    assert normalized_to_screen(0.1, 0.1) == (1.0, 1.0)
    # 1 normalised → 10 px on both axes
    assert normalized_to_screen(1, 1) == (10.0, 10.0)


def test_round_trip():
    """screen → normalised → screen should be the identity."""
    for sx, sy in [(0, 0), (100, 400), (600, 400), (1100, 725)]:
        nx, ny = screen_to_normalized(sx, sy)
        sx2, sy2 = normalized_to_screen(nx, ny)
        assert abs(sx2 - sx) < 1e-9
        assert abs(sy2 - sy) < 1e-9

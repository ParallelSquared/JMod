import polars as pl

# Import the functions we want to test
from src.utils.io.read_output import (
    find_extrema_in_nearby_scans, calculate_peak_smoothness
)

class TestFindExtremaNearbyScans:
    """Minimal coverage tests for find_extrema_in_nearby_scans."""

    def test_basic_run(self):
        # Tiny 3-scan mock dataset
        df = pl.DataFrame({
            "seq": ["A", "A", "A"],
            "z": [2, 2, 2],
            "rt": [1.0, 2.0, 3.0],
            "coeff": [10.0, 20.0, 15.0],
            "intensity": [100, 200, 150],
        })

        out = find_extrema_in_nearby_scans(
            df,
            column_names=["intensity"],
            find_max_list=[True],
            n_scans=1
        )

        # Should add nearby max column
        assert "intensity_nearby_max" in out.columns



class TestCalculatePeakSmoothness:
    def test_minimal(self):
        # Minimal 3-point peak
        df = pl.DataFrame({
            "seq": ["A", "A", "A"],
            "z": [2, 2, 2],
            "rt": [1.0, 2.0, 3.0],
            "coeff": [10.0, 20.0, 15.0],
            "n_scans": [3, 3, 3],
        })

        out = calculate_peak_smoothness(df)

        # Has smoothness column
        assert "smoothness" in out.columns

        # No null smoothness values
        assert out["smoothness"].is_null().sum() == 0

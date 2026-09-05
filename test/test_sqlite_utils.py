"""Tests for comfort.util.sqlite_utils."""
import sqlite3
import numpy as np
import pytest

from comfort.util.sqlite_utils import Median, Std, validate_identifier


class TestValidateIdentifier:
    # Accept a normal table/column name
    def test_accepts_valid_identifier(self):
        validate_identifier("P_NITRATE")  # Must not raise

    # Accept identifiers starting with an underscore
    def test_accepts_underscore_prefix(self):
        validate_identifier("_hidden")

    # Reject a name containing a SQL injection payload
    def test_rejects_sql_injection_attempt(self):
        with pytest.raises(ValueError):
            validate_identifier("foo; DROP TABLE bar;--")

    # Reject non-string input
    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            validate_identifier(123)

    # Reject identifiers that start with a digit
    def test_rejects_identifier_starting_with_digit(self):
        with pytest.raises(ValueError):
            validate_identifier("1table")


@pytest.fixture
def conn_with_values():
    conn = sqlite3.connect(":memory:")
    conn.create_aggregate("median", 1, Median)
    conn.create_aggregate("std", 1, Std)
    conn.execute("CREATE TABLE t (v REAL)")
    yield conn
    conn.close()


class TestMedianAggregate:
    # Median of an odd-length column
    def test_odd_count(self, conn_with_values):
        conn_with_values.executemany("INSERT INTO t VALUES (?)", [(1.0,), (3.0,), (2.0,)])
        result = conn_with_values.execute("SELECT median(v) FROM t").fetchone()[0]
        assert result == pytest.approx(2.0)

    # Result matches numpy's median for an even-length column
    def test_matches_numpy_median(self, conn_with_values):
        vals = [5.0, 1.0, 9.0, 3.0]
        conn_with_values.executemany("INSERT INTO t VALUES (?)", [(v,) for v in vals])
        result = conn_with_values.execute("SELECT median(v) FROM t").fetchone()[0]
        assert result == pytest.approx(np.median(vals))


class TestStdAggregate:
    # Result matches numpy's sample standard deviation (ddof=1)
    def test_matches_numpy_sample_std(self, conn_with_values):
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        conn_with_values.executemany("INSERT INTO t VALUES (?)", [(v,) for v in vals])
        result = conn_with_values.execute("SELECT std(v) FROM t").fetchone()[0]
        assert result == pytest.approx(np.std(vals, ddof=1))

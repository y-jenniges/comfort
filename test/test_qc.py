"""Tests for comfort.qc."""
import logging
import pandas as pd
import pytest

from comfort.qc import (QC_ALL, QC_GOOD, QCFilter, apply_qc_flags,
                        build_where_clause, flag_salinity_like_oxygen)


class TestBuildWhereClause:
    # No flags means no WHERE clause at all
    def test_empty_flags_returns_empty(self):
        assert build_where_clause([]) == ""

    # None is treated the same as an empty list
    def test_none_flags_returns_empty(self):
        assert build_where_clause(None) == ""

    # A single filter renders as a plain WHERE
    def test_single_flag(self):
        flags = [QCFilter("PQF1", ">0")]
        assert build_where_clause(flags) == "WHERE PQF1>0"

    # Multiple filters are AND-joined
    def test_multiple_flags(self):
        result = build_where_clause(QC_GOOD)
        assert "WHERE" in result
        assert "PQF1>0" in result
        assert "PQF2>2" in result
        assert "AND" in result

    # table_alias prefixes every column
    def test_table_alias(self):
        flags = [QCFilter("PQF1", ">0")]
        result = build_where_clause(flags, table_alias="t")
        assert result == "WHERE t.PQF1>0"

    # keyword="AND" is used for appending onto an existing WHERE clause
    def test_keyword_and(self):
        flags = [QCFilter("PQF1", ">0")]
        result = build_where_clause(flags, keyword="AND")
        assert result == "AND PQF1>0"

    # table_alias and keyword combine for a multi-table join filter
    def test_alias_and_keyword_combined(self):
        flags = [QCFilter("PQF1", ">0"), QCFilter("PQF2", ">2")]
        result = build_where_clause(flags, table_alias="q", keyword="AND")
        assert result.startswith("AND ")
        assert "q.PQF1>0" in result
        assert "q.PQF2>2" in result

    # Plain tuples work the same as QCFilter tuples
    def test_tuple_input(self):
        flags = [("PQF1", ">0"), ("PQF2", ">2")]
        result = build_where_clause(flags)
        assert "PQF1>0" in result
        assert "PQF2>2" in result

    # QC_ALL is the no-filtering preset
    def test_qc_all_is_empty(self):
        assert build_where_clause(QC_ALL) == ""


class TestApplyQcFlags:
    @staticmethod
    def _df():
        # Test df
        return pd.DataFrame({
            "PQF1": [0, 1, 1, 1],
            "PQF2": [0, 1, 3, 5],
            "VAL": [10.0, 20.0, 30.0, 40.0],
        })

    # quality_flags=None passes every row through
    def test_none_returns_copy(self):
        df = self._df()
        result = apply_qc_flags(df, None)
        assert len(result) == 4
        assert result is not df

    # An empty filter list also passes every row through
    def test_empty_returns_copy(self):
        df = self._df()
        result = apply_qc_flags(df, [])
        assert len(result) == 4

    # ">" operator string is parsed and applied
    def test_gt_operator(self):
        result = apply_qc_flags(self._df(), [QCFilter("PQF1", ">0")])
        assert (result["PQF1"] > 0).all()
        assert len(result) == 3

    # ">=" operator string is parsed and applied
    def test_ge_operator(self):
        result = apply_qc_flags(self._df(), [QCFilter("PQF2", ">=3")])
        assert (result["PQF2"] >= 3).all()
        assert len(result) == 2

    # "<" operator string is parsed and applied
    def test_lt_operator(self):
        result = apply_qc_flags(self._df(), [QCFilter("PQF2", "<3")])
        assert (result["PQF2"] < 3).all()

    # "<=" operator string is parsed and applied
    def test_le_operator(self):
        result = apply_qc_flags(self._df(), [QCFilter("PQF2", "<=3")])
        assert (result["PQF2"] <= 3).all()

    # "==" operator string is parsed and applied
    def test_eq_operator(self):
        result = apply_qc_flags(self._df(), [QCFilter("PQF1", "==1")])
        assert (result["PQF1"] == 1).all()

    # "!=" operator string is parsed and applied
    def test_ne_operator(self):
        result = apply_qc_flags(self._df(), [QCFilter("PQF1", "!=0")])
        assert (result["PQF1"] != 0).all()

    # The QC_GOOD preset combines both PQF1 and PQF2 thresholds
    def test_qc_good_preset(self):
        # QC_GOOD means PQF1 > 0 AND PQF2 > 2
        result = apply_qc_flags(self._df(), QC_GOOD)
        assert len(result) == 2
        assert (result["PQF1"] > 0).all()
        assert (result["PQF2"] > 2).all()

    # A filter on a column absent from df warns and leaves rows untouched
    def test_missing_column_warns(self, caplog):
        df = self._df()
        with caplog.at_level(logging.WARNING):
            result = apply_qc_flags(df, [QCFilter("SQF", ">0")])
        assert "SQF" in caplog.text
        # All rows returned when the filter column is absent
        assert len(result) == 4

    # The returned DataFrame is independent of the input
    def test_returns_copy(self):
        df = self._df()
        result = apply_qc_flags(df, QC_GOOD)
        result["VAL"] = 999.0
        assert df["VAL"].iloc[0] != 999.0

    # Plain tuples work the same as QCFilter tuples
    def test_tuple_input(self):
        result = apply_qc_flags(self._df(), [("PQF1", ">0")])
        assert (result["PQF1"] > 0).all()


class TestFlagSalinityLikeOxygen:
    _SP_TO_SA = 35.16504 / 35

    def _df(self):
        """Two genuine oxygen rows and one salinity value mislabelled as oxygen."""
        sp = 35.0
        return pd.DataFrame({
            "OXYGEN": [250.0, 180.0, sp],
            "SALINITY": [sp, 34.5, sp],
        })

    # Only the row where OXYGEN numerically matches SALINITY is flagged
    def test_flags_mislabelled_row_only(self, caplog):
        with caplog.at_level(logging.WARNING):
            mask = flag_salinity_like_oxygen(self._df())
        assert list(mask) == [False, False, True]
        assert "mislabelled salinity" in caplog.text

    # SA (Absolute Salinity) is used instead of Practical Salinity when present
    def test_sa_column_preferred_over_salinity(self):
        df = self._df()
        df["SA"] = df["SALINITY"] * self._SP_TO_SA
        mask = flag_salinity_like_oxygen(df)
        assert list(mask) == [False, False, True]

    # Column detection is case-insensitive
    def test_lowercase_salinity_column(self):
        df = self._df().rename(columns={"SALINITY": "salinity"})
        mask = flag_salinity_like_oxygen(df)
        assert list(mask) == [False, False, True]

    # salinity_col lets the caller point at a non-standard column name
    def test_explicit_salinity_col(self):
        df = self._df().rename(columns={"SALINITY": "PSAL"})
        mask = flag_salinity_like_oxygen(df, salinity_col="PSAL")
        assert list(mask) == [False, False, True]

    # A NaN salinity value can never match, so the row is not flagged
    def test_nan_rows_not_flagged(self):
        df = self._df()
        df.loc[2, "SALINITY"] = float("nan")
        mask = flag_salinity_like_oxygen(df)
        assert not mask.any()

    # OXYGEN is a required column
    def test_missing_oxygen_column_raises(self):
        with pytest.raises(ValueError, match="OXYGEN"):
            flag_salinity_like_oxygen(pd.DataFrame({"SALINITY": [35.0]}))

    # A salinity column (SA or SALINITY) is required
    def test_missing_salinity_column_raises(self):
        with pytest.raises(ValueError, match="SALINITY"):
            flag_salinity_like_oxygen(pd.DataFrame({"OXYGEN": [250.0]}))

    # min_diff/max_diff narrow or widen the matching range
    def test_custom_range(self):
        df = self._df()
        mask = flag_salinity_like_oxygen(df, min_diff=1.0, max_diff=2.0)
        assert not mask.any()

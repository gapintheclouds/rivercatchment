"""Tests for statistics functions within the Model layer."""

import pandas as pd
import pandas.testing as pdt
import datetime
import pytest


@pytest.mark.parametrize(
    "test_data, test_index, test_columns, expected_data, expected_index, expected_columns",
    [
        (
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [pd.to_datetime('2000-01-01 01:00'),
                 pd.to_datetime('2000-01-01 02:00'),
                 pd.to_datetime('2000-01-01 03:00')],
                ['A', 'B'],
                [[0.0, 0.0]],
                [datetime.date(2000, 1, 1)],
                ['A', 'B']
        ),
        (
                [[1, 2], [3, 4], [5, 6]],
                [pd.to_datetime('2000-01-01 01:00'),
                 pd.to_datetime('2000-01-01 02:00'),
                 pd.to_datetime('2000-01-01 03:00')],
                ['A', 'B'],
                [[3.0, 4.0]],
                [datetime.date(2000, 1, 1)],
                ['A', 'B']
        )
    ]
)
def test_daily_mean(test_data, test_index, test_columns,
                    expected_data, expected_index, expected_columns):
    """Test mean function works with zeros and positive integers"""
    from catchment.models import daily_mean
    pdt.assert_frame_equal(
        daily_mean(pd.DataFrame(data=test_data, index=test_index, columns=test_columns)),
        pd.DataFrame(data=expected_data, index=expected_index, columns=expected_columns))


@pytest.mark.parametrize(
    "test_data, test_index, test_columns, expected_data, expected_index, expected_columns",
    [
        (
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [pd.to_datetime('2000-01-01 01:00'),
                 pd.to_datetime('2000-01-01 02:00'),
                 pd.to_datetime('2000-01-01 03:00')],
                ['A', 'B'],
                [[0.0, 0.0]],
                [datetime.date(2000, 1, 1)],
                ['A', 'B']
        ),
        (
                [[1, 2], [3, 4], [5, 6]],
                [pd.to_datetime('2000-01-01 01:00'),
                 pd.to_datetime('2000-01-01 02:00'),
                 pd.to_datetime('2000-01-01 03:00')],
                ['A', 'B'],
                [[5, 6]],
                [datetime.date(2000, 1, 1)],
                ['A', 'B']
        )
    ]
)
def test_daily_max(test_data, test_index, test_columns,
                   expected_data, expected_index, expected_columns):
    """Test mean function works with zeros and positive integers"""
    from catchment.models import daily_max
    pdt.assert_frame_equal(
        daily_max(pd.DataFrame(data=test_data, index=test_index, columns=test_columns)),
        pd.DataFrame(data=expected_data, index=expected_index, columns=expected_columns))


@pytest.mark.parametrize(
    "test_data, test_index, test_columns, expected_data, expected_index, expected_columns",
    [
        (
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [pd.to_datetime('2000-01-01 01:00'),
                 pd.to_datetime('2000-01-01 02:00'),
                 pd.to_datetime('2000-01-01 03:00')],
                ['A', 'B'],
                [[0.0, 0.0]],
                [datetime.date(2000, 1, 1)],
                ['A', 'B']
        ),
        (
                [[1, 2], [3, 4], [5, 6]],
                [pd.to_datetime('2000-01-01 01:00'),
                 pd.to_datetime('2000-01-01 02:00'),
                 pd.to_datetime('2000-01-01 03:00')],
                ['A', 'B'],
                [[1, 2]],
                [datetime.date(2000, 1, 1)],
                ['A', 'B']
        )
    ]
)
def test_daily_min(test_data, test_index, test_columns,
                   expected_data, expected_index, expected_columns):
    """Test mean function works with zeros and positive integers"""
    from catchment.models import daily_min
    pdt.assert_frame_equal(
        daily_min(pd.DataFrame(data=test_data, index=test_index, columns=test_columns)),
        pd.DataFrame(data=expected_data, index=expected_index, columns=expected_columns))


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from catchment.models import daily_mean

    test_input = pd.DataFrame(
        data=[[0.0, 0.0],
              [0.0, 0.0],
              [0.0, 0.0]],
        index=[pd.to_datetime('2000-01-01 01:00'),
               pd.to_datetime('2000-01-01 02:00'),
               pd.to_datetime('2000-01-01 03:00')],
        columns=['A', 'B']
    )
    test_result = pd.DataFrame(
        data=[[0.0, 0.0]],
        index=[datetime.date(2000, 1, 1)],
        columns=['A', 'B']
    )

    # Need to use Pandas testing functions to compare arrays
    pdt.assert_frame_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from catchment.models import daily_mean

    test_input = pd.DataFrame(
        data=[[1, 2],
              [3, 4],
              [5, 6]],
        index=[pd.to_datetime('2000-01-01 01:00'),
               pd.to_datetime('2000-01-01 02:00'),
               pd.to_datetime('2000-01-01 03:00')],
        columns=['A', 'B']
    )
    test_result = pd.DataFrame(
        data=[[3.0, 4.0]],
        index=[datetime.date(2000, 1, 1)],
        columns=['A', 'B']
    )

    # Need to use Pandas testing functions to compare arrays
    pdt.assert_frame_equal(daily_mean(test_input), test_result)


def test_daily_min_integers():
    """Test that mean function works for an array of positive integers."""
    from catchment.models import daily_min

    test_input = pd.DataFrame(
        data=[[1, 2],
              [3, 4],
              [5, 6]],
        index=[pd.to_datetime('2000-01-01 01:00'),
               pd.to_datetime('2000-01-01 02:00'),
               pd.to_datetime('2000-01-01 03:00')],
        columns=['A', 'B']
    )
    test_result = pd.DataFrame(
        data=[[1, 2]],
        index=[datetime.date(2000, 1, 1)],
        columns=['A', 'B']
    )

    # Need to use Pandas testing functions to compare arrays
    pdt.assert_frame_equal(daily_min(test_input), test_result)


def test_daily_max_integers():
    """Test that mean function works for an array of positive integers."""
    from catchment.models import daily_max

    test_input = pd.DataFrame(
        data=[[1, 2],
              [3, 4],
              [5, 6]],
        index=[pd.to_datetime('2000-01-01 01:00'),
               pd.to_datetime('2000-01-01 02:00'),
               pd.to_datetime('2000-01-01 03:00')],
        columns=['A', 'B']
    )
    test_result = pd.DataFrame(
        data=[[5, 6]],
        index=[datetime.date(2000, 1, 1)],
        columns=['A', 'B']
    )

    # Need to use Pandas testing functions to compare arrays
    pdt.assert_frame_equal(daily_max(test_input), test_result)

import pytest
import pandas as pd
import numpy as np
import datetime
from ts_avg import MultiseasonalAveraging
import itertools

@pytest.fixture()
def seasonal_list():
    intraday_coeff = [1,0,0,0]
    intraweek_coeff = np.ones(7)
    weekly_coeff = [1, 1, 1]
    return [intraday_coeff, intraweek_coeff, weekly_coeff]

@pytest.fixture()
def seasonal_nplist(seasonal_list):
    return [np.array(arr) for arr in seasonal_list]

@pytest.fixture()
def seasonal_list_decay_week():
    intraday_coeff = [1,0,0,0]
    intraweek_coeff = np.ones(7)
    weekly_coeff = [0.25, 0.5, 1]
    return [intraday_coeff, intraweek_coeff, weekly_coeff]

@pytest.fixture()
def feb_ts_df():
    start_date = pd.to_datetime('2/1/2023')
    end_date = pd.to_datetime('3/1/2023')
    date_series = pd.date_range(start=start_date, end=end_date, freq=pd.to_timedelta('6h'), closed='left')
    df = pd.DataFrame({'date': date_series})
    df['y'] = df.index
    df['y'] = df['y'].apply(lambda x: 1 + np.cos(0.05*(x%4)) + 0.025*((x//4)%7))
    return df

@pytest.fixture()
def const_ts_df():
    start_date = pd.to_datetime('2/1/2023')
    end_date = pd.to_datetime('3/1/2023')
    date_series = pd.date_range(start=start_date, end=end_date, freq=pd.to_timedelta('6h'), closed='left')
    df = pd.DataFrame({'ds': date_series})
    df['val'] = 1.0
    return df

@pytest.fixture()
def linear_increase_daily_period_weekly():
    start_date = pd.to_datetime('2/1/2023')
    end_date = pd.to_datetime('3/1/2023')
    date_series = pd.date_range(start=start_date, end=end_date, freq=pd.to_timedelta('6h'), closed='left')
    df = pd.DataFrame({'date': date_series})
    df['y'] = df.index
    df['y'] = df['y'].apply(lambda x: (x%4)*((x//4)%7))
    return df

# test 1
def test_outer_flatten_1(seasonal_list, seasonal_nplist):
    calculated = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                           1.0, 0.0, 0.0, 0.0])
    
    # from a list of lists/np.arrays
    from_function = MultiseasonalAveraging.outer_flatten(seasonal_list)
    np.testing.assert_array_equal(from_function, calculated)

    # from a list of np.arrays
    from_function = MultiseasonalAveraging.outer_flatten(seasonal_nplist)
    np.testing.assert_array_equal(from_function, calculated)


# test 2
def test_outer_flatten_2(seasonal_list_decay_week):
    calculated = np.array([0.25, 0.00, 0.00, 0.00, 0.25, 0.00, 0.00, 0.00, 0.25,
                           0.00, 0.00, 0.00, 0.25, 0.00, 0.00, 0.00, 0.25, 0.00,
                           0.00, 0.00, 0.25, 0.00, 0.00, 0.00, 0.25, 0.00, 0.00,
                           0.00, 0.50, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00,
                           0.50, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.50,
                           0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.50, 0.00,
                           0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00,
                           0.00, 1.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00,
                           1.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 1.00,
                           0.00, 0.00, 0.00])
    
    from_function = MultiseasonalAveraging.outer_flatten(seasonal_list_decay_week)
    np.testing.assert_array_equal(from_function, calculated)

# test for coefficients 1

def test_coefficients():
    n = 3
    array_length = 10
    calculated = np.ones(array_length) / float(array_length)
    from_function = MultiseasonalAveraging.coefficients(n, array_length, 'ones')
    np.testing.assert_array_equal(from_function, calculated)

    calculated = np.zeros(array_length)
    calculated[n] = 1.0
    from_function = MultiseasonalAveraging.coefficients(n, array_length, 'self')
    np.testing.assert_array_equal(from_function, calculated)

    calculated = np.array([np.e**(-i) for i in range(array_length-1, -1, -1)])
    calculated = calculated / np.sum(calculated)
    from_function = MultiseasonalAveraging.coefficients(n, array_length, 'exponential_decay')
    np.testing.assert_array_almost_equal(from_function, calculated)


# test averaging 1
def test_averaging(feb_ts_df):
    '''
    An arbitrary time series for 28 days with 4 windows per day
    The prediction for the next 28 days must be identical to
    the give time series
    '''
    msa = MultiseasonalAveraging(feb_ts_df, date='date', y='y')
    seasonal_dict_list = [{'period': 4, 'function': 'self'},
                          {'period': 28, 'function': 'self'},
                          {'period': 112, 'function': 'self'}]
    msa.get_averages(seasonal_dict_list, 112, 'test')
    avg_df = msa.avg_df_list[0]['avg_df']
    y = feb_ts_df['y'].to_numpy()
    yhat = avg_df['yhat'].to_numpy()

    np.testing.assert_array_almost_equal(y, yhat)

    # check if the starting index of the predicted forecast
    # is a continuation of the supplied df
    assert avg_df.index.start == feb_ts_df.index.stop

def test_date_from_timestep(feb_ts_df):
    msa = MultiseasonalAveraging(feb_ts_df, date='date', y='y')
    # 1
    calculated = pd.to_datetime('2/2/2023 18:00:00')
    from_function = msa.get_date_from_timestep(7)
    assert abs(from_function - calculated) < pd.to_timedelta('1ms')
    # 2
    calculated = pd.to_datetime('1/30/2023 12:00:00')
    from_function = msa.get_date_from_timestep(-6)
    assert abs(from_function - calculated) < pd.to_timedelta('1ms')

def test_avg_on_flatdata(const_ts_df):
    '''
    In this test predictions will be made on a time series that takes
    a constant value (1.0). Irrespective of the averaging method used, the
    values for every forecast window must be constant (1.0)
    '''
    # checking arbitrary column label names
    msa = MultiseasonalAveraging(const_ts_df, date='ds', y='val')
    expected_avg = 1.0
    expected_std = 0.0
        
    possible_functions = ['ones', 'self', 'exponential_decay']
    # generating all possible permutations of three options above
    n = 3
    permutation_list = set(itertools.permutations(possible_functions*n, n))

    for i, functions in enumerate(permutation_list):
        seasonal_dict_list = [{'period': 4, 'function': functions[0]},
                              {'period': 28, 'function': functions[1]},
                              {'period': 112, 'function': functions[2]}
                            ]
        msa.get_averages(seasonal_dict_list, 112, 'test')
        avg_df = msa.avg_df_list[i]['avg_df']

        # average must be 1 irrespective of the method of averaging
        yhat = avg_df['yhat'].to_numpy()
        np.testing.assert_array_almost_equal(expected_avg, yhat)

        # standard deviation must be zero, again, irrespective of the method
        # of averaging
        ystd = avg_df['ystd'].to_numpy()
        np.testing.assert_array_almost_equal(expected_std, ystd)

def test_linear_increase_daily_period_weekly_1(linear_increase_daily_period_weekly):
    seasonal_dict_list = [{'period': 4, 'function': 'self'},
                          {'period': 28, 'function': 'self'},
                          {'period': 112, 'function': 'exponential_decay'},
                          ]

    msa = MultiseasonalAveraging(linear_increase_daily_period_weekly)
    msa.get_averages(seasonal_dict_list, 112, 'test')
    # check yhat
    yhat = msa.avg_df_list[0]['avg_df']['yhat'].to_numpy()
    yhat_expected = linear_increase_daily_period_weekly['y'].to_numpy()
    np.testing.assert_array_almost_equal(yhat, yhat_expected)

    # check ystd
    ystd_expected = 0.0
    ystd = msa.avg_df_list[0]['avg_df']['ystd'].to_numpy()
    np.testing.assert_array_almost_equal(ystd, ystd_expected)

def test_linear_increase_daily_period_weekly_2(linear_increase_daily_period_weekly):
    seasonal_dict_list = [{'period': 4, 'function': 'ones'},
                          {'period': 28, 'function': 'self'},
                          {'period': 112, 'function': 'exponential_decay'},
                          ]

    msa = MultiseasonalAveraging(linear_increase_daily_period_weekly)
    msa.get_averages(seasonal_dict_list, 112, 'test')
    # check yhat
    yhat = msa.avg_df_list[0]['avg_df']['yhat'].reset_index(drop=True).to_numpy()
    # 0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, ... for one month
    yhat_expected = np.outer(np.ones(4), np.outer(np.arange(7)*1.5, np.ones(4)).flatten()).flatten()
    np.testing.assert_array_almost_equal(yhat, yhat_expected)

    # check ystd
    ystd_expected = np.outer(np.ones(4), (np.outer((np.std(np.arange(4)) * np.arange(7)), np.ones(4)).flatten())).flatten()
    ystd = msa.avg_df_list[0]['avg_df']['ystd'].to_numpy()
    np.testing.assert_array_almost_equal(ystd, ystd_expected)







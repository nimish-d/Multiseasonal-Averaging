import pytest
import pandas as pd
import numpy as np
from ts_avg import MultiseasonalAveraging

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
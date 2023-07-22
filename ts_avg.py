import numpy as np
import pandas as pd


class MultiseasonalAveraging():
    def __init__(self, df, date='date', y='y'):
        self.date = date
        self.y = y
        self.df = df[[self.date, self.y]].copy(deep=True)
        self.df.reset_index(inplace=True) # index retained
        self.get_ntimesteps()
        self.get_timewindow()
    
    def get_ntimesteps(self):
        self.step_start = self.df.index.start
        self.step_end = self.df.index.stop
        if (self.step_start == self.step_end):
            raise ValueError(f'The supplied dataframe has only one timestep')
        self.index_start = self.df.iloc[0]['index']
        self.index_end = self.df.iloc[-1]['index']
        self.ntimesteps = self.step_end - self.step_start
        self.start_time = self.df.iloc[0][self.date]

    def get_timewindow(self):
        df = self.df
        date = self.date
        dt_timedelta = (df.iloc[1][date] - df.iloc[0][date])
        dt_seconds = dt_timedelta.total_seconds()
        dt_seconds_array = (df[date].iloc[1:].reset_index(drop=True)
                    - df[date].iloc[0:-1].reset_index(drop=True)).dt.total_seconds().to_numpy()
        if (not np.all(np.isclose(dt_seconds, dt_seconds_array))):
            raise ValueError('The time intervals between consecutive entries of the supplied dataframe are not equal')
        else:
            self.dt_timedelta = dt_timedelta

    @staticmethod
    def outer_flatten(seasonal_list):
        outer = np.ones(1)
        for i in range(len(seasonal_list)):
            outer = np.outer(np.array(seasonal_list[i]), outer).flatten()
        return outer
    
    @staticmethod
    def coefficients(n, period, function='ones'):
        if(function == 'ones'):
            arr = np.ones(period)
        elif(function == 'self'):
            arr = np.zeros(period)
            arr[n] = 1.0
        elif(function == 'exponential_decay'):
            arr = np.exp(np.arange(period))
        else:
            print('Incorrect function supplied')
            return None
        arr = arr / np.sum(arr)
        return arr

    def get_date_from_index(self, idx):
        return self.start_time + self.dt_timedelta * (idx - self.index_start)

    def __append_to_average_table(self):
        try:
            idx = len(self.avg_df_list)
        except:
            self.avg_df_list = []
            idx = 0
        finally:
            self.avg_df_list.append({})
        return idx

    def get_averages(self, seasonal_dict_list, ntimesteps_forecast=None, name=None):
        date = self.date
        y = self.y
        
        # Check if the total number of timesteps available in the supplied dataframe
        # is adequate for the averaging
        len_full_coefficients = np.product([adict['period'] for adict in seasonal_dict_list])
        if (len_full_coefficients < self.ntimesteps):
            raise ValueError(f'''The number of timesteps ({self.ntimesteps}) in the
                             supplied time series dataframe is fewer than the
                             number of coefficients required for seasonal averaging
                             ({len_full_coefficients})''')

        nseasonality = len(seasonal_dict_list)
        
        # check if seasonality is integral multiple of the previous seasonality
        # of the multiseasonality ts
        seasonality_list = [adict['period'] for adict in seasonal_dict_list]
        div_seasonality_list = [seasonality_list[0]]
        for i in range(nseasonality-1):
            if ((seasonality_list[i+1] % seasonality_list[i]) != 0):
                raise ValueError('a period of mutiseasonality is not an integer multiple of the previous peroid')
            else:
                div_seasonality_list.append(int(seasonality_list[i+1] // seasonality_list[i]))

        # create a dictionary element in the avg_df_list to add details of the
        # average, std, etc..
        idx = self.__append_to_average_table()
        self.avg_df_list[idx]['name'] = name
        self.avg_df_list[idx]['seasonal_dict_list'] = seasonal_dict_list
        self.avg_df_list[idx]['nseasonality'] = nseasonality
        self.avg_df_list[idx]['seasonality_list'] = seasonality_list
        self.avg_df_list[idx]['div_seasonality_list'] = div_seasonality_list
        df_extract = self.df.iloc[-len_full_coefficients:]
        self.avg_df_list[idx]['df_extract'] = df_extract      

        # Caclulation of averages, std. deviation etc.
        # Given a new time step, calculate `n` for each
        # of the multiseasonality
        self.avg_df_list[idx]['step_forecast_start'] = df_extract.index.stop
        self.avg_df_list[idx]['step_forecast_end'] = self.avg_df_list[idx]['step_forecast_start'] + ntimesteps_forecast

        y_history = df_extract[y].to_numpy()
        avg_list = []
        std_list = []
        for ts in range(self.avg_df_list[idx]['step_forecast_start'], self.avg_df_list[idx]['step_forecast_end']):
            dividend = ts
            coefficients_list = []
            for i in range(nseasonality):
                dividend, n = divmod(dividend, div_seasonality_list[i])
                c = MultiseasonalAveraging.coefficients(n, div_seasonality_list[i], seasonal_dict_list[i]['function'])
                coefficients_list.append(c)
            coefficients = MultiseasonalAveraging.outer_flatten(coefficients_list)
            # calculate the weighted average of the observations
            avg = np.dot(coefficients, y_history)
            non_zero_coeff = np.logical_not(np.isclose(np.abs(coefficients), 0.0, atol=1.0e-15))
            std = np.std(y_history[non_zero_coeff])
            avg_list.append(avg)
            std_list.append(std)
        avg_df =  pd.DataFrame({'yhat': avg_list, 'ystd': std_list})                                                                                              
        avg_df.set_index(keys=pd.Index(range(self.avg_df_list[idx]['step_forecast_start'], self.avg_df_list[idx]['step_forecast_end'])),
                         inplace=True)

        self.avg_df_list[idx]['avg_df'] = avg_df
        # Adding date/time information to dataframe
        self.avg_df_list[idx]['avg_df'][date] = pd.Series(self.avg_df_list[idx]['avg_df'].index).apply(lambda id: self.get_date_from_index(id)).values
        self.avg_df_list[idx]['avg_df'] = self.avg_df_list[idx]['avg_df'].reindex(columns=[date, 'yhat', 'ystd'])

                                          


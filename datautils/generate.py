import pandas as pd
import pandas_datareader.data as web
from os import path, makedirs
import matplotlib.pyplot as plt
from datetime import datetime, date 
import sys
import requests 
import datetime 
import re
# ...

# ####### Reading interest rate information ####### #
interest_rates = staticmethod(lambda sampling, start, end:\
                 web.DataReader('DGS10', 'fred', start, end).resample(sampling).last().div(100).dropna()) 

# ######## get SP500 index benchmark data ########## #
get_sp500 = lambda start = '01-01-2010', end = date.today(): web.DataReader(['sp500'], 'fred', start, end)

# ###################################### #
# SP500: class to access and process     #
#   SP500 companies                      # 
# ###################################### #

class SP500:
    """
    class to pull sp500 list of companies and to perform analytics 
        on individual performances
    """

    def __init__(self, save_path = None,  **iex_info):
        data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        data = data[['Symbol', 'GICS Sector']]
        data.columns = ['ticker', 'sector']
        self.company_list = data
        self.sectors = [name.split(' ')[0] for name in self.company_list['sector'].tolist()]
        self.tickers = self.company_list['ticker'].tolist()
        self.sector_map = {key:name for key,name in zip(self.company_list['sector'].tolist(), self.sectors)}
        self._save_path = None
        if save_path is not None:
            file_name = 'SP500_Symbols_On_' + date.today().strftime('%b-%d-%Y') + '.csv' 
            data.to_csv(path.join(save_path, file_name), sep = ',', header=True, index=True)
            self._save_path = save_path
        self._sp_assets = None 
    
    @property
    def sp_assets(self):
        return self._sp_assets
    
    @sp_assets.setter
    def asset_frame(self, new_frame):
        self._sp_assets = new_frame 
    
    @property
    def save_path(self):
        return self._save_path 

    @save_path.setter
    def save_path(self, new_path):
        self._save_path = new_path     
    
    @property
    def index_price(self):
        return get_sp500() 

    @property
    def performance_returns(self):
        return [attr for attr in self.__dict__ if 'returns' in attr]  

    def sector_weights(self, return_type = ['fig']):
        """
        shows the weights of SP500 based on sector
        the weight is shown based on the number of tickers in ech sector
        """
        data = self.company_list.groupby('sector').count().sort_values(by='ticker')
        fig, axs = plt.subplots(figsize = (8,6))
        plt.rcParams.update({'font.size': 30})
        plt.rcParams.update({'font.family': 'times'})    
        axs.bar(data.index, data['ticker'].values/sum(data['ticker'].values))
        axs.set_xlabel('sector')
        axs.set_ylabel('share in sp500 portfolio')
        plt.xticks(rotation = 'vertical')
        returns = []
        if 'data' in return_type:
            returns.append(data)
        if 'fig' in return_type: 
            returns.extend([fig, axs])
        else:
            return None 
        return returns 

    def performance(self, sector='Information Technology', history = '2y', load_data = None, 
                        sampling = 'M', sort_by = 'mean', data_source = 'yahoo'):
        """
        generate asset returns for each sector or for the whole sp500 list
        the purpose is to indetify assets in each sector with highest performance
        sort_by: 'mean' or 'volatility' reports a sorted frame of asset performance
        """
        if self._sp_assets is None:
            if load_data is not None:
                self._sp_assets = pd.read_csv(load_data, delimiter=',', header = 0).apply(lambda col:
                                 pd.to_datetime(col) if col.name == 'Date' else col).set_index('Date')
            else:
                self._sp_assets = {'yahoo': stock_history_yahoo}[data_source](self.tickers, history = history,
                    save_path = self._save_path)
                self.compare_tickers(self._sp_assets.columns)
        if sector is not None:
            """
            returns for sector tickers
            """
            tickers = self.company_list[self.company_list['sector'] == sector]['ticker'].tolist() 
            sector_returns = self._sp_assets[tickers].resample(sampling).last().pct_change().dropna()
            attr_name = self.sector_map[sector].lower() + '_returns'
            setattr(self, attr_name, sector_returns)
            sorted_df = {'mean': SP500.sort_by_mean, 
                            'volatility': SP500.sort_by_std}[sort_by](sector_returns)
        else:
            """
            overall returns
            """
            sp_returns = self._sp_assets.resample(sampling).last().pct_change().dropna()
            attr_name = 'sp500_returns'
            setattr(self, attr_name, sp_returns)
            sorted_df = {'mean': SP500.sort_by_mean, 
                            'volatility': SP500.sort_by_std}[sort_by](sp_returns)            
        return sorted_df
    
    # useful utility methods 
    sort_by_mean = staticmethod(lambda return_frame: return_frame.mean(0).sort_values(axis=0))
    sort_by_std = staticmethod(lambda return_frame: return_frame.std(0).sort_values(axis=0))

    def compare_tickers(self, tickers):
        diff = set(tickers)^set(self.tickers)
        if len(diff) != 0:
            print('the following tickers are missed in data pull >>> ',  list(diff))
        else:
            print('history for all requested tickers were obtained during data pull ...')

# ########################################## #
# IEXCloud: to access and process historical  
#       data from IEXCloud.io                 
# ########################################## #

class IEXCloud:
    pulls = 0
    def __init__(self, tickers = None, token = None, data_type='sandbox'):
        if '.csv' in tickers:
            self.tickers = pd.read_csv(tickers)['Ticker'].tolist()
        else:
            self.tickers = tickers
        
        if token != None:
            self.token = token
        else:
            print('it is not possible to pull data without a Token; exit')
            sys.exit(1) 
        
        self.data_type= data_type
    
    def pull_stock_history_batch(self, num_chunks = 20, history = '5y',
                return_df=True, concat_axis = 0):
        IEXCloud.pulls += 1
        print('pull #: ', IEXCloud.pulls)
        print('pull type: sandbox or real? ', self.data_type)
        print('This will take a while ... ')    
        
        if len(self.tickers) > num_chunks:
            sub_len = len(self.tickers)//num_chunks 
            symbol_group = [self.tickers[n_chunk:(n_chunk + 1)*sub_len] for n_chunk in range(num_chunks)]
            string_groups = [','.join(symbol) for symbol in symbol_group]
        else:
            string_groups = [','.join(self.tickers)] 
        all_data = []
        for group in string_groups:
            batch_req = {'sandbox': f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={group}&types=chart&range={history}&token={self.token}',
                            'real': f'https://cloud.iexapis.com/stable/stock/market/batch?symbols={group}&types=chart&range={history}&token={self.token}'}[self.data_type]
            data = requests.get(batch_req, verify = True)

            if data.status_code != 200:
                print('IEX data access error due to code ', data.status_code)
                sys.exit(-1)            
            data = data.json()
            all_data.extend([pd.DataFrame.from_dict(data[ticker]['chart']).set_index(['date']).rename(columns={'symbol':'ticker'})
                for ticker in group.split(',') if ticker in data.keys() and ticker != None])
        all_data = pd.concat(all_data, axis = concat_axis)
        all_data.index = pd.to_datetime(all_data.index)
        if return_df:
            return all_data 
    
    @staticmethod
    def _to_hdf(data_df, save_path, file_name, key):
        file_name += '.h5'
        if not path.exists(save_path):
            makedirs(save_path)
        data_df.to_hdf(path.join(save_path, file_name), key)
    
    @staticmethod
    def _pass(*args, **kwargs):
        pass 


# ################################################### #
# ##########        Useful functions      ########### #
# ################################################### #
#. 1) pull batch data from iex cloud

def stock_history_from_iex(tickers = None, token = None,
         data_type = 'sandbox',
                 history = '5y',
                  use_columns = 'close', save_path = None):
    today = date.today()
    use_columns = [use_columns]
    if token == None:
        print('It is not possible to pull the data without a token, exit')
        sys.exit(1)
    _iex = IEXCloud(tickers = tickers,
         token = token, data_type = data_type)
    asset_frame = _iex.pull_stock_history_batch(concat_axis = 1,
          history = history)
    use_columns = use_columns + ['ticker']
    asset_frame = asset_frame[use_columns]
    new_names = asset_frame['ticker'].dropna().drop_duplicates().values.tolist()[0]
    asset_frame = asset_frame.drop(['ticker'], axis = 1)
    asset_frame.columns = new_names
    if save_path:
        save_name = 'Stock_History_' + data_type + '_PriceHandle_' + use_columns[0].upper() + '_Pull_Count_' +\
                str(_iex.pulls) + '_For_' + str(len(tickers)) + '_Tickers_on_' + today.strftime('%b-%d-%Y') + '.csv' 
        #tools.save_to_hdf(asset_frame, save_path, save_name, 'IEX')    
        asset_frame.to_csv(save_name, sep=' ', header = True, index = True, float_format = '%.5f')
    return asset_frame, new_names

# ####### Reading Stock history from yahoo finance using pandas datareader ####### #

def stock_history_yahoo(tickers = None, history = '2y', use_columns = 'Close', save_path = None):
    
    print('pulling data from yahoo finance ... ')
    num = re.findall('\d+', history)[0]
    hist_type = re.findall('[a-zA-Z]', history)[0]
    delta_ = {'d':1, 'm':30, 'y':365}[hist_type]
    now = datetime.datetime.today()
    delta = datetime.timedelta(days = int(num)*delta_)
    start = now - delta 
    start = start.strftime('%Y-%m-%d')
    end = now.strftime('%Y-%m-%d')
    asset_frame = []
    for tick in tickers:
        try:
            _frame = web.DataReader(tick, 'yahoo', start = start, end = end).rename({use_columns:tick}, axis = 1, inplace=False)[tick]
        except:
            pass
        else:
            asset_frame.append(_frame)            
    hist = pd.concat(asset_frame, axis = 1).dropna()
    if save_path is not None:
        save_name = 'Stock_History_Yahoo_Finance_' +str(len(tickers)) + '_Tickers_from_' + start + '_to_' + end + '.csv'
        hist.to_csv(path.join(save_path, save_name), header=True, index=True, float_format='%.5f')
    return hist 

# ####### Reading ETF prices using Datareader ####### #
def etf_price_history(etfs = None, source = 'yahoo', date_range = ('01-01-2010', date.today()), use_column = 'Close'):
    if etfs is None:
        etfs = ['SPY', 'QQQ', 'BND', 'BBH']
    start, end = date_range 
    etf = web.DataReader(etfs, source, start = start, end = end)
    idx = pd.IndexSlice 
    etf = etf.loc[:, idx[use_column, :]]
    etf.columns = etf.columns.droplevel(0)
    return etf 







import pandas as pd
import pandas_datareader.data as web
from os import path, listdir, makedirs, getcwd 
from collections.abc import Iterable
from natsort import natsorted
from timeit import default_timer
from functools import wraps
import matplotlib
import gzip 
import shutil 
from struct import unpack 
from collections import namedtuple 
import matplotlib.pyplot as plt
from datetime import datetime, date 
import sys
import requests 
import datetime 
# ...
from .. utils import tools 


# ###################################### #
# AlgoSeek: class to access and process  #
#   AlgoSeek data files.                 #
# ###################################### #

class AlgoSeekCSV:
    """
    Class for processing AlgoSeek csv files of Nasdaq 100 tickers
    DataFrame object is wrapped in the class and different methods
        are added for quantitative analysis of data
    The main data attribute is 'seek_df' which is a dataframe itself.
    AlgoSeekCSV class can be used for building a machine learning model as a data parser class 
    """
    tcols = ['openbartime', 'firsttradetime', 'highbidtime', 'highasktime',
         'hightradetime', 'lowbidtime', 'lowasktime', 'lowtradetime', 'closebartime','lasttradetime']

    drop_cols = ['unknowntickvolume', 'cancelsize', 'tradeatcrossorlocked']

    keep = ['firsttradeprice', 'hightradeprice', 'lowtradeprice', 'lasttradeprice',
                'minspread', 'maxspread', 'volumeweightprice', 'nbboquotecount',
            'tradeatbid', 'tradeatbidmid', 'tradeatmid', 'tradeatmidask',
                'tradeatask', 'volume', 'totaltrades', 'finravolume', 'finravolumeweightprice', 'uptickvolume', 'downtickvolume',
        'repeatuptickvolume', 'repeatdowntickvolume', 'tradetomidvolweight', 'tradetomidvolweightrelative']

    columns = {'volumeweightprice': 'price',
           'finravolume': 'fvolume',
           'finravolumeweightprice': 'fprice',
           'uptickvolume': 'up',
           'downtickvolume': 'down',
           'repeatuptickvolume': 'rup',
           'repeatdowntickvolume': 'rdown',
           'firsttradeprice': 'first',
           'hightradeprice': 'high',
           'lowtradeprice': 'low',
           'lasttradeprice': 'last',
           'nbboquotecount': 'nbbo',
           'totaltrades': 'ntrades',
           'openbidprice': 'obprice',
           'openbidsize': 'obsize',
           'openaskprice': 'oaprice',
           'openasksize': 'oasize',
           'highbidprice': 'hbprice',
           'highbidsize': 'hbsize',
           'highaskprice': 'haprice',
           'highasksize': 'hasize',
           'lowbidprice': 'lbprice',
           'lowbidsize': 'lbsize',
           'lowaskprice': 'laprice',
           'lowasksize': 'lasize',
           'closebidprice': 'cbprice',
           'closebidsize': 'cbsize',
           'closeaskprice': 'caprice',
           'closeasksize': 'casize',
           'firsttradesize': 'firstsize',
           'hightradesize': 'highsize',
           'lowtradesize': 'lowsize',
           'lasttradesize': 'lastsize',
           'tradetomidvolweight': 'volweight',
           'tradetomidvolweightrelative': 'volweightrel'}

    def __init__(self, csv_data):
        self.seek_df = csv_data
    
    @staticmethod
    def _to_hdf(data_path, key_csv, key):
        save_path = path.join(data_path, key + '_Processed')
        if not path.exists(save_path):
            makedirs(save_path)
        key_csv.to_hdf(path.join(save_path, 'algoseek.h5'), 'min_sorted')
    
    @staticmethod
    def _pass(*args):
        pass

    @classmethod
    def from_gz_files(cls, data_path = None, output = 'hdf5'):
        print('this is going to take a while ...')
        start = default_timer()
        main_dirs = [_dir for _dir in listdir(data_path) if '.DS_Store' not in _dir and 'Processed' not in _dir]
        _paths = [path.join(data_path, _dir) for _dir in main_dirs]
        csv_files = {name:[] for name in main_dirs}
        for key,_path in zip(csv_files.keys(), _paths):
            dirs = natsorted([_dir for _dir in listdir(_path) if '.DS_Store' not in _dir], key=lambda y: y.lower())
            for _dir in dirs:
                csv_files[key].extend([path.join(_path, _dir, _file) for _file in listdir(path.join(_path, _dir))])
        
        for key in csv_files.keys():
            key_csv = []
            for _file in csv_files[key]:
                key_csv.append(pd.read_csv(_file, parse_dates=[['Date', 'TimeBarStart']])
                    .rename(columns=str.lower)
                    .drop(cls.tcols + cls.drop_cols, axis=1)
                    .rename(columns=cls.columns)
                    .set_index('date_timebarstart')
                    .sort_index()
                    .between_time('9:30', '16:00')
                    .set_index('ticker', append=True)
                    .swaplevel()
                    .rename(columns=lambda x: x.replace('tradeat', 'at')))
            key_csv = pd.concat(key_csv).apply(pd.to_numeric, downcast = 'integer')
            key_csv.index.rename(['ticker', 'date_time'], inplace = True)
            {'hdf5': tools.save_to_hdf,
                 'none': AlgoSeekCSV._pass}[output](key_csv, data_path, 'algoSeek.h5', key)
        end = default_timer()
        print('dataframe object seek_df is generated from csv files in ', end - start, ' seconds...!')
        return cls(key_csv)
            
    @classmethod
    def from_hdf(cls, data_path = None):
        return cls(pd.read_hdf(data_path))
        
    # helpers
    def counter_all(func):
        @wraps(func)
        def func_wrapper(self, *args, **kwargs):
            func_wrapper.counter += 1
            func(self, *args, **kwargs)
        func_wrapper.counter = 0
        return func_wrapper
    
    @counter_all
    def column_help(self, key = None):
        if self.column_help.counter == 1:
            self.column_keys = {item:key for key,item in self.columns}
        print('the longer format of the key is ...')
        if key == None:
            return self.column_keys
        else:
            return self.column_keys[key]
    
    # process and data visualization
    def plot_ticker_day(self, date = None, ticker = None, what = None):
        idx = pd.IndexSlice
        single_date = True
        if isinstance(date, Iterable):
            begin, end =  datetime.strptime(date[0], '%Y-%m-%d'), datetime.strptime(date[1], '%Y-%m-%d')
            single_date = False
        else:
            date = datetime.strptime(date, '%Y-m-%d')

        fig, axs = plt.subplots(figsize=(10,8))
        if single_date:
            self.seek_df.loc[idx[ticker]][self.seek_df.loc[idx[ticker]].index.date == date.date()].plot(y = what, kind= 'line', ax=axs, use_index=True, legend = True, linewidth = 2, color = 'black')
        else:
            self.seek_df.loc[idx[ticker]][(self.seek_df.loc[idx[ticker]].index.date >= begin.date()) & 
                (self.seek_df.loc[idx[ticker]].index.date <= end.date())].plot(y = what, kind = 'line', ax=axs, use_index=True, legend = True, linewidth = 2, color = 'black')
        return fig, axs

# ########################################## #
# Nasdaq ITCH data : process and parse data                 
# ########################################## #

class ITCH:
    """
    class to parse nasdaq itch message files
    """
    event_codes = {'O': 'Start of Messages',
                    'S': 'Start of System Hours',
                    'Q': 'Start of Market Hours',
                    'M': 'End of Market Hours',
                    'E': 'End of System Hours',
                    'C': 'End of Messages'}
    encoding = {'primary_market_maker': {'Y': 1, 'N': 0},
                'printable'           : {'Y': 1, 'N': 0},
                'buy_sell_indicator'  : {'B': 1, 'S': -1},
                 'cross_type'          : {'O': 0, 'C': 1, 'H': 2},
                'imbalance_direction' : {'B': 0, 'S': 1, 'N': 0, 'O': -1}}
    formats = {('integer', 2): 'H', ('integer', 4): 'I', ('integer', 6): '6s', 
                 ('integer', 8): 'Q', ('alpha', 1)  : 's', ('alpha', 2)  : '2s', ('alpha', 4)  : '4s',
                ('alpha', 8)  : '8s', ('price_4', 4): 'I', ('price_8', 8): 'Q'}
    
    def __init__(self, file_path, file_name, message_path = None, message_name = 'message_types.xlsx'):
        self.itch_file = None
        self.itch_store = None
        self.alpha_formats = None
        self.alpha_length = None
        self.message_fields = {}
        self.fstrings = {}

        if not message_path:
            _message_path = path.join(file_path, 'helpers', message_name)
        else:
            _message_path = path.join(message_path, message_name)
        self.message_types = ITCH.clean_message(pd.read_excel(_message_path,sheet_name= 'messages').sort_values('id').drop('id', axis=1))
        self.read_itch_file(file_path, file_name)
        self._generate_message_types()
        self._generate_alphanumerics()
        self._generate_msg_fields_strings()


    def _generate_message_types(self):
        self.message_labels = self.message_types[['message_type','notes']].dropna().rename(columns={'notes':'name'})
        self.message_labels['name'] = self.message_labels['name'].str.lower().str.replace('message','').str.replace('.','').str.strip().str.replace(' ','_')
        self.message_types['message_type'] = self.message_types['message_type'].ffill()
        self.message_types['message_type'] = self.message_types[self.message_types['name'] != 'message_type']
        self.message_types['value'] = self.message_type['value'].str.lower().str.replace(' ','_').str.replace('(','').replace(')','')
        self.message_types['formats'] = self.message_types[['value','length']].apply(tuple, axis=1).map(self.formats)    

    def _generate_alphanumerics(self):
        alpha_fields = self.message_types[self.message_types['value'] == 'alpha'].set_index('name')
        alpha_msg = alpha_fields.groupby('message_type')
        self.alpha_formats = {key:val.to_dict() for key,val in alpha_msg['formats']}
        self.alpha_length = {key:val.add(5).to_dict() for key,val in alpha_msg['length']}
    
    def _generate_msg_fields_strings(self):
        for t, message in self.message_types.groupby('message_type'):
            self.message_fields[t] = namedtuple(typename=t, field_names = message.name.tolist())
            self.fstrings[t] = '>' + ''.join(message.formats.tolist())
    
    def format_alphanumerics(self, msg_type, data):
        for col in self.alpha_formats.get(msg_type).keys():
            if msg_type != 'R' and col == 'stock':
                data = data.drop(col, axis=1)
                continue
            data[col] = data[col].str.decode('utf-8').str.strip()
            if self.encoding.get(col):
                data[col] = data[col].map(self.encoding.get(col))
        return data
    

    def read_itch_file(self, file_path, file_name):
        gz_file = path.join(file_path, file_name)
        filename = ''.join(file_name.split('.')[:2]) + '.bin'
        self.itch_store = path.join(file_path, filename, '_itch_store.h5') 
        self.itch_file = path.join(file_path, filename)
        if not path.exists(self.itch_file):
            try:
                with gzip.open(gz_file, 'rb') as f_in:
                    with open(self.itch_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except EOFError:
                print('itch file unzipped and stored in a binary file ...', self.itch_file)
    
    @staticmethod
    def clean_message(df):
        df.columns = [col.lower().strip() for col in df.columns]
        df['value'] = df['value'].str.strip()
        df['name'] = (df['name'].str.strip().str.lower().str.replace(' ','_').str.replace('-','_').str.replace('/','_'))
        df['notes'] = df['notes'].str.strip()
        df['message_types'] = df.loc[df['name'] == 'message_type', 'value']
        return df
    
    @classmethod
    def from_itch(cls, file_path, file_name, message_path = None, message_name = 'message_types.xlsx'):
        return cls(file_path, file_name, message_path, message_name)

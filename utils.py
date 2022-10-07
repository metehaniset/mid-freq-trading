import pandas as pd
import numpy as np
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, accuracy_score

from machine_learning.lib.features import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import math

"""
ITCH generation, parsing
cat 20200804/TR-* 20200805/TR-* 20200806/TR-* 20200807/TR-* ... > 20200804_20200811.itch
bin/BookConstructor /home/hft/data/binary/20200804_20200811.itch /home/hft/data/book/ /home/hft/data/messages/ 10 HEKTS.E
"""
def get_orderbook(stock="GARAN", resample_period="500L", max_index="2020-08-07", read_from_pickle=True):
    """
    L       milliseonds
    U       microseconds
    S       seconds
    """
    filename = '/home/hft/20200804/book/' + 'TR-3.VRD_' + stock + '.E_book_10.csv'
    filename = '/home/hft/data/book/' + '20200804_20200811.itch_' + stock + '.E_book_10.csv'

    try:
        if read_from_pickle:
            df = pd.read_pickle('/home/hft/data/pickle/orderbook_' + stock + "-" +
                                str(resample_period) + "-" + str(max_index) + ".pickle")
            print(stock, 'data readed from PICKLE')
            return df
    except:
        pass
    print(stock, 'data calculating from CSV')
    df = pd.read_csv(filename, dtype={'time_ns': object})
    print('len(df)', len(df))
    df['time'] = pd.to_datetime(df['time'], unit="s")
    df["date"] = pd.to_datetime(df["time"].dt.strftime('%Y-%m-%d %H:%M:%S') + '.' + df["time_ns"])
    df.drop(['time', 'time_ns'], axis=1, inplace=True)
    df.set_index('date', inplace=True)
    df = df.sort_index()
    df = df.loc[df.index <= max_index]
    df = df.between_time('07:00:00', '14:59:59')

    if resample_period is not None:
        resampled = df.groupby([df.index.day]).resample(resample_period).last()
        df = resampled.reset_index(level=[0], drop=True)
    else:
        print('YOU SHOULD RESAMPLE AT LEAST 250L. Exiting now')
        sys.exit()

    ms = get_messagebook(stock=stock, resample_period=resample_period, max_index=max_index)
    df = pd.concat([df, ms], axis=1, sort=True)

    # print(df.head(100))
    # df = resampled.reindex(df.index).fillna(method='bfill')
    # print(df.head(1000).to_string())
    # sys.exit()
    # df = df.groupby(pd.TimeGrouper('1D'))
    # print(df.head(10).to_string())
    # sys.exit()
    print('Orderbook len(df) after resample and getting times betwen 07:00-15:00', len(df))
    # sys.exit()
    # df = df.resample('1L')
    df.to_pickle('/home/hft/data/pickle/orderbook_' + stock + "-" +
                 str(resample_period) + "-" + str(max_index) + ".pickle")

    return df


def get_messagebook(stock="GARAN", max_index="2020-08-05", resample_period="500L", columns=['time', 'time_ns', 'side', 'size']):
    #  filename = '/home/hft/20200804/messages/' + 'TR-3.VRD_' + stock + '.E_message.csv'
    filename = '/home/hft/data/messages/' + '20200804_20200811.itch_' + stock + '.E_message.csv'

    print(stock, 'data calculating from CSV')
    df = pd.read_csv(filename, dtype={'time_ns': object}, usecols=columns)
    print('len(df)', len(df))
    df['time'] = pd.to_datetime(df['time'], unit="s")
    df["date"] = pd.to_datetime(df["time"].dt.strftime('%Y-%m-%d %H:%M:%S') + '.' + df["time_ns"])
    df.drop(['time', 'time_ns'], axis=1, inplace=True)
    df.set_index('date', inplace=True)
    df = df.sort_index()
    df = df.loc[df.index <= max_index]
    df = df.between_time('07:00:00', '14:59:59')
    # resampled = df.resample(resample_period)
    # print(df.head(100).to_string())
    # resampled = df.groupby([df.index.day, 'side']).resample(resample_period).sum()   # agg({'side': 'last', 'size': 'sum'})
    # print(resampled.head(100))
    # sys.exit()
    # df = resampled.reset_index(level=[0], drop=True)
    # print(df)
    # sys.exit()
    # df['1_rate_qty'] = df['side'].rolling(arrival_rate_period, min_periods=1).apply(lambda x: df.loc[x.loc[x == 1].index]['size'].sum(), raw=False)
    # print('here')

    # def myfunc(x, side):
    #     y = df.loc[x.index]  # .reset_index()
    #     _sum = y.loc[y['side'] == side]['size'].sum()
    #     try:
    #         print(x.tail(1).index.item(), _sum)
    #     except:
    #         pass
    #     return _sum
    resampled = pd.DataFrame()
    print(resample_period, 'Resample message data started: ', datetime.now())
    resampled['side_1_vol'] = df['side'].resample(resample_period).apply(lambda x:  df.loc[x.loc[x == 1].index]['size'].sum())
    print('Resampling first column completed:', datetime.now())
    resampled['side_0_vol'] = df['side'].resample(resample_period).apply(lambda x:  df.loc[x.loc[x == 0].index]['size'].sum())
    resampled['side_1_cnt'] = df['side'].resample(resample_period).apply(lambda x: len(x.loc[x == 1]))
    resampled['side_0_cnt'] = df['side'].resample(resample_period).apply(lambda x: len(x.loc[x == 0]))
    print('All resampling completed: ', datetime.now())
    # Calculate rates
    for second in [1, 5, 10, 30, 60, 300, 600, 1800]:
        resampled['side_1_cumvol_'+str(second)+'sec'] = resampled['side_1_vol'].rolling(str(second)+"s", min_periods=1).sum()
        resampled['side_0_cumvol_'+str(second)+'sec'] = resampled['side_0_vol'].rolling(str(second)+"s", min_periods=1).sum()
        resampled['side_1_cumcnt_'+str(second)+'sec'] = resampled['side_1_cnt'].rolling(str(second)+"s", min_periods=1).sum()
        resampled['side_0_cumcnt_'+str(second)+'sec'] = resampled['side_0_cnt'].rolling(str(second)+"s", min_periods=1).sum()


    # df['all_rate_qty'] = df['side'].rolling(arrival_rate_period, min_periods=1).count()
    # print('all_rate_qty')
    # print('1_rate_qty', datetime.now())
    # df['1_rate_qty'] = df['side'].rolling(arrival_rate_period, min_periods=1).apply(lambda x: myfunc(x, 1), raw=False)# (lambda x: df.loc[x.index].loc[x == 1]['size'].sum(), raw=False)
    # print('1_rate_qty', datetime.now())
    # df['0_rate_qty'] = df['side'].rolling(arrival_rate_period, min_periods=1).apply(lambda x: myfunc(x, 0), raw=False) # (lambda x: df.loc[x.index].loc[x == 0]['size'].sum(), raw=False)
    # print('0_rate_qty', datetime.now())

    # df['1_rate_cnt'] = df['side'].rolling(arrival_rate_period, min_periods=1).apply(lambda x: len(x.loc[x == 1]))
    # df['0_rate_cnt'] = df['side'].rolling(arrival_rate_period, min_periods=1).apply(lambda x: len(x.loc[x == 0]))
    # df['all_rate_cnt'] = df['side'].rolling(arrival_rate_period, min_periods=1).count()

    print('Messagebook len(df) after resample and getting times betwen 07:00-15:00', len(df))
    return resampled

# get_messagebook(stock="GUBRF")



# def get_last_value_of_each_second():
#     ind = features.index
#     resampled = features['mean_price'].groupby([ind.year, ind.month, ind.day, ind.hour, ind.minute, ind.second]).apply(
#         lambda x: x.iloc[[-1]])
#     resampled.reset_index(level=[0, 1, 2, 3, 4, 5], drop=True, inplace=True)
#     resampled = resampled.reindex(df.index).fillna(method='bfill')
#     # resampled = resampled.reindex(features.index)
#     print(resampled.head(10000).to_string())
#     print(resampled.value_counts())
#     sys.exit()

def perform_grid_search(X_data, y_data):
    parameters = {'max_depth': [2, 3, 4, 5],  # 10, 20],
                  'n_estimators': [1, 10, 25, 50, 100],  # , 1024, 2048],
                  'random_state': [42]}
    rf = RandomForestClassifier(criterion='gini', class_weight='balanced_subsample')
    clf = GridSearchCV(rf, parameters, cv=4, scoring='neg_log_loss', n_jobs=6)  # neg_log_loss, accuracy
    clf.fit(X_data, y_data)
    print(clf.cv_results_['mean_test_score'])
    return clf.best_params_['n_estimators'], clf.best_params_['max_depth']

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, accuracy_score

from machine_learning.lib.features import *
from machine_learning.lib.utils import drop_unnecessary_features
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from hft.lib.utils import get_orderbook, get_messagebook, perform_grid_search
import math

"""
ITCH parsing
cat 20200804/TR-* 20200805/TR-* 20200806/TR-* 20200807/TR-* ... > 20200804_20200811.itch
bin/BookConstructor /home/hft/data/binary/20200804_20200811.itch /home/hft/data/book/ /home/hft/data/messages/ 10 HEKTS.E


http://eugenezhulenev.com/blog/2014/11/14/stock-price-prediction-with-big-data-and-machine-learning/

"""
"""
L       milliseonds
U       microseconds
S       seconds
"""
df = get_orderbook(stock="GARAN", resample_period='60S', max_index="2020-08-12", read_from_pickle=True)
# print(df.head(10).to_string())
# sys.exit()
"""
#
#   Trend labeling, tripple barier labeling denenecek. SVM rbf kernel denenecek
#
"""
# mb = get_messagebook(stock="GARAN", resample_period=resample_period, columns=['time', 'time_ns', 'side'])


features = pd.DataFrame(index=df.index)
features['1_price_mean'] = (df['1_ask_price'] + df['1_bid_price'])/2
# features['mean_volume'] = (features['1_ask_vol'] + features['1_bid_vol'])/2
# features['bid_ask_spread'] = (features['1_ask_price'] - features['1_bid_price'])
features['total_volume_dif'] = df.filter(regex='_ask_vol$', axis=1).sum(axis=1) - df.filter(regex='_bid_vol$', axis=1).sum(axis=1)
max_allowed_depth = 3
drop_columns = []
for i in range(max_allowed_depth+1, 11):    # bunları düşürüyorum ki bütün derinlikleri işin içine kattığım hesaplar dengeli olsun
    drop_columns.append(str(i)+'_ask_price')
    drop_columns.append(str(i) + '_bid_price')
    drop_columns.append(str(i) + '_ask_vol')
    drop_columns.append(str(i) + '_bid_vol')
if len(drop_columns) > 0:
    df.drop(columns=drop_columns, inplace=True)

for i in range(1, max_allowed_depth):
    # Basic set
    features[str(i) + '_ask_price'] = df[str(i) + '_ask_price']
    features[str(i) + '_bid_price'] = df[str(i) + '_bid_price']
    features[str(i) + '_ask_vol'] = df[str(i) + '_ask_vol']
    features[str(i) + '_bid_vol'] = df[str(i) + '_bid_vol']

    #time insensitive
    features[str(i) + '_askbid_price_dif'] = df[str(i) + '_ask_price'] - df[str(i) + '_bid_price']  # bid-ask spread and mid price
    features[str(i) + '_askbid_price_mean'] = (df[str(i) + '_ask_price'] + df[str(i) + '_bid_price']) / 2

    try:
        features[str(i) + '_ask_step'] = abs(df[str(i+1) + '_ask_price'] - df[str(i) + '_ask_price'])  # price differences
        features[str(i) + '_bid_step'] = abs(df[str(i + 1) + '_bid_price'] - df[str(i) + '_bid_price'])
    except:
        pass
    # accumulated differences


    features[str(i) + '_askbid_vol_dif'] = df[str(i) + '_ask_vol'] - df[str(i) + '_bid_vol']
    # features[str(i) + '_price_mean'] = (df[str(i) + '_ask_price'] + df[str(i) + '_bid_price'])/2
    features[str(i) + '_volume_mean'] = (df[str(i) + '_ask_vol'] + df[str(i) + '_bid_vol']) / 2
    features[str(i) + '_askvol_mul'] = df[str(i) + '_ask_vol'] / df.filter(regex='_ask_vol$', axis=1).sum(axis=1)   # ask_volume/total_ask_volume
    features[str(i) + '_bidvol_mul'] = df[str(i) + '_bid_vol'] / df.filter(regex='_bid_vol$', axis=1).sum(axis=1)
    features[str(i) + '_askbidvol_mul'] = df[str(i) + '_ask_vol'] / df.filter(regex='_bid_vol$', axis=1).sum(axis=1)  # ask_volume/total_bid_volume
    features[str(i) + '_bidaskvol_mul'] = df[str(i) + '_bid_vol'] / df.filter(regex='_ask_vol$', axis=1).sum(axis=1)

    # Tavana ya da tabana yakınlığını da hesaplayabilirim aslında. Binary içinde var bu veri de

# en üstle kademe ve ilk kademe arasındaki fark
features['top_bottom_ask_price_dif'] = df[str(max_allowed_depth) + '_ask_price'] - df['1_ask_price']
features['top_bottom_bid_price_dif'] = df['1_bid_price'] - df[str(max_allowed_depth) + '_bid_price']
# mean prices and volumes
features['ask_prices_mean'] = df.filter(regex='_ask_price$', axis=1).sum(axis=1)/max_allowed_depth
features['bid_prices_mean'] = df.filter(regex='_bid_price$', axis=1).sum(axis=1)/max_allowed_depth
features['ask_vol_mean'] = df.filter(regex='_ask_vol$', axis=1).sum(axis=1)/max_allowed_depth
features['bid_vol_mean'] = df.filter(regex='_bid_vol$', axis=1).sum(axis=1)/max_allowed_depth
features['accumulated_price_dif'] = features.filter(regex='_askbid_price_dif$', axis=1).sum(axis=1)
features['accumulated_vol_dif'] = features.filter(regex='_askbid_vol_dif$', axis=1).sum(axis=1)

features['side_1_cumvol_1sec'] = df['side_1_cumvol_300sec']
features['side_0_cumvol_1sec'] = df['side_0_cumvol_300sec']
features['side_1_cumcnt_1sec'] = df['side_1_cumcnt_300sec']
features['side_0_cumcnt_1sec'] = df['side_0_cumcnt_300sec']

features['side_1_cumvol_5sec'] = df['side_1_cumvol_600sec']
features['side_0_cumvol_5sec'] = df['side_0_cumvol_600sec']
features['side_1_cumcnt_5sec'] = df['side_1_cumcnt_600sec']
features['side_0_cumcnt_5sec'] = df['side_0_cumcnt_600sec']

outcomes = pd.DataFrame(index=df.index)
outcomes['next_20_bar'] = features['1_price_mean'].shift(-1).pct_change().apply(np.sign)     # 5L'lik resamplingte yaklaşık 1 saniyeye denk geliyor
outcomes['next_50_bar'] = features['1_price_mean'].shift(-20).pct_change().apply(np.sign)
outcomes['next_100_bar'] = features['1_price_mean'].shift(-100).pct_change().apply(np.sign)
# outcomes['next_1_sec'] = \

# print(outcomes.to_string())
print(outcomes['next_20_bar'].value_counts())
outcome = outcomes['next_20_bar']


features = features.dropna()
outcome = outcome.dropna()
features = features.reindex(outcome.index)
features.dropna(inplace=True)
outcome = outcome.reindex(features.index)

features = drop_unnecessary_features(features, outcome, threshold=0.50)
print(features.columns)

X_train = features["2020-08-04":"2020-08-10"]
y_train = outcome["2020-08-04":"2020-08-10"]

X_validate = features["2020-08-11":"2020-08-12"]
y_validate = outcome["2020-08-11":"2020-08-12"]

#
# X_train = features.reindex(y_train.index)
# X_train.dropna(inplace=True)  # drop N/A
# y_train = y_train.reindex(X_train.index)


# extract parameters
n_estimator, depth = perform_grid_search(X_train, y_train)
print(n_estimator, depth)
# n_estimator = 128
# depth = 5

# Refit a new model with best params, so we can see feature importance
rf = RandomForestClassifier(max_depth=depth, n_estimators=n_estimator,
                            criterion='gini', random_state=42, class_weight='balanced_subsample')

# rf = DecisionTreeClassifier(max_depth=depth,
#                             criterion='gini', random_state=42, class_weight=None)

rf.fit(X_train, y_train.values.ravel())

"""
Training metrics
"""
# Performance Metrics
# y_pred_rf = rf.predict_proba(X_train)[:, 1]
y_pred = rf.predict(X_train)
# fpr_rf, tpr_rf, _ = roc_curve(y_train, y_pred_rf)
print('TRAININ METRICS')
print(classification_report(y_train, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_train, y_pred))

print('')
print("Accuracy")
print(accuracy_score(y_train, y_pred))


y_pred = rf.predict(X_validate)
print('VALIDATION METRICS')
print(classification_report(y_validate, y_pred))
print("Confusion Matrix")
print(confusion_matrix(y_validate, y_pred))
print('')
print("Accuracy")
print(accuracy_score(y_validate, y_pred))


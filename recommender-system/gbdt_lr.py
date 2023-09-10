from utils import creat_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import lightgbm as lgb

"""
gbdt+lr细节详解 https://zhongqiang.blog.csdn.net/article/details/108349729
gbdt 数据从跟节点到叶子节点的路径，相当于各种特征组合的路径
lr 因为系数惩罚不会过拟合，能处理高纬度的稀疏特征
"""
#
df_train, _, dense_feats, sparse_feats = creat_dataset(5000)
for f in sparse_feats:
	tmp = pd.get_dummies(df_train[f], prefix=f)
	df_train = df_train.drop(f, axis=1)
	df_train = pd.concat([df_train, tmp], axis=1)
train = df_train[[col for col in df_train if col != 'label']]
label = df_train['label']
x_train, x_val, y_train, y_val = train_test_split(train, label, test_size=0.2, random_state=2022)


#%%
lr = LogisticRegression()
lr.fit(x_train, y_train)
tr_loss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
val_loss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
print(tr_loss, val_loss)
# 0.16254190232812632 0.5538221883627097


#%%
gbm = lgb.LGBMClassifier(
	objective='binary',
	subsample=0.8,
	colsample_bytree=0.7,
	learning_rate=0.01,
	n_estimators=10000
)
gbm.fit(
	x_train, y_train,
	eval_set=[(x_train, y_train), (x_val, y_val)],
	eval_names=['train', 'val'],
	eval_metric='binary_logloss',
	early_stopping_rounds=100
)
tr_loss = log_loss(y_train, gbm.predict_proba(x_train)[:, 1])
val_loss = log_loss(y_val, gbm.predict_proba(x_val)[:, 1])
print(tr_loss, val_loss)
# 0.36353428637754964 0.49624374227656615


#%%
gbm = lgb.LGBMClassifier(
	objective='binary',
	subsample=0.8,
	colsample_bytree=0.7,
	learning_rate=0.01,
	n_estimators=10000
)
gbm.fit(
	x_train, y_train,
	eval_set=[(x_train, y_train), (x_val, y_val)],
	eval_names=['train', 'val'],
	eval_metric='binary_logloss',
	early_stopping_rounds=100
)
model = gbm.booster_
# 每个样本落入每棵树叶节点的INDEX,n_booster就是gbdt early_stopping时的总树数
# (n_samples, n_booster)
gbm_feats_train = model.predict(train, pred_leaf=True)
gbm_feats_names = ['gbm_leaf_' + str(i) for i in range(gbm_feats_train.shape[1])]
gbm_feats_train = pd.DataFrame(gbm_feats_train, columns=gbm_feats_names)
train = pd.concat([train, gbm_feats_train], axis=1)
# 叶节点的INDEX做one_hot,lr模型因为正则化不会对分类变量过拟合
for f in gbm_feats_names:
	tmp = pd.get_dummies(train[f], prefix=f)
	train = train.drop(f, axis=1)
	train = pd.concat([train, tmp], axis=1)
x_train, x_val, y_train, y_val = train_test_split(train, label, test_size=0.2, random_state=2021)
lr = LogisticRegression()
lr.fit(x_train, y_train)
tr_loss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
val_loss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
print(tr_loss, val_loss)
# 0.01478348667526691 0.39278386104776725




















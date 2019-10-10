# from the 17th solution in IEEE-CIS Fraud Detection
# https://www.kaggle.com/c/ieee-fraud-detection/discussion/111696#latest-644676
# 对于树方法，传入best_iteration,然后进行全量数据的predict，这时候是没有val 的值输出的
# 这个best_iteration 是在80% 的数据训练上产出的， 那么对于后面全量数据有什么不一样的地方？
# 即这时候的validation 的没有意义？
def make_test_prediction(X, y, X_test, best_iteration, seed=SEED, category_cols=None):
    print('best iteration:', best_iteration)
    preds = np.zeros((X_test.shape[0], NFOLDS))

    print(X.shape, X_test.shape)
    
    skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
    
    for i, (trn_idx, _) in enumerate(skf.split(X, y)):
        fold = i + 1
        print('Fold:',fold)
        
        tr_x, tr_y = X.iloc[trn_idx,:], y.iloc[trn_idx]
            
        print(len(tr_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)

        clf = lgb.train(lgb_params, tr_data, best_iteration, categorical_feature=category_cols)
        preds[:, i] = clf.predict(X_test)
    
    return preds

# 这里做validation 的时候，采用的是三个seed ，然后每个做cv
# 这里并没有做cv， 而是把这个数据当做了validatino的数据，在这个效果上最好
# 然后设置early- stoping， 剔除seed造成的扰动
# params['seed'] = s， params['bagging_seed'] = s， params['feature_fraction_seed'] = s
        
def make_val_prediction(X_train, y_train, X_val, y_val, seed=SEED, seed_range=3, lgb_params=lgb_params,
                        category_cols=None):
    print(X_train.shape, X_val.shape)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    auc_arr = []
    best_iteration_arr = []
    val_preds = np.zeros((X_val.shape[0], seed_range))
    
    feature_importance_df = pd.DataFrame()
    feature_importance_df['feature'] = X_train.columns.tolist()
    feature_importance_df['gain_importance'] = 0
    feature_importance_df['split_importance'] = 0
    
    for i, s in enumerate(range(seed, seed + seed_range)):
        seed_everything(s)
        params = lgb_params.copy()
        params['seed'] = s
        params['bagging_seed'] = s
        params['feature_fraction_seed'] = s

        clf = lgb.train(params, train_data, 10000, valid_sets = [train_data, val_data], categorical_feature=category_cols,
                        early_stopping_rounds=500, feval=eval_auc, verbose_eval=200)

        best_iteration = clf.best_iteration
        best_iteration_arr.append(best_iteration)
        val_pred = clf.predict(X_val, best_iteration)
        val_preds[:, i] = val_pred
        
        auc = fast_auc(y_val, val_pred)
        auc_arr.append(auc)
        print('seed:', s, ', auc:', auc, ', best_iteration:', best_iteration)

        feature_importance_df['gain_importance'] += clf.feature_importance('gain')/seed_range
        feature_importance_df['split_importance'] += clf.feature_importance('split')/seed_range

    auc_arr = np.array(auc_arr)
    best_iteration_arr = np.array(best_iteration_arr)
    best_iteration = int(np.mean(best_iteration_arr))

    avg_pred_auc = fast_auc(y_val, np.mean(val_preds, axis=1))
    print(f'avg pred auc: {avg_pred_auc:.5f}, avg auc: {np.mean(auc_arr):.5f}+/-{np.std(auc_arr):.5f}, avg best iteration: {best_iteration}')

    feature_importance_df = feature_importance_df.sort_values(by='split_importance', ascending=False).reset_index(drop=True)
    plot_feature_importance(feature_importance_df)
    display(feature_importance_df.head(20))
    
    return feature_importance_df, best_iteration, val_preds


# 第一部，分层hold-out  20%的数据量 
X_train, y_train, X_val, y_val = train_val_split_by_time(X, y)

# 第二部， 进行处理，然后产出最优的树的迭代次数， 这个是根据val 的数据上的效果反馈
# 这里的分层的hold-out 的数据是用来做validation 来用的。产出最好的迭代次数
X_train, X_val, category_cols1 = fe1(X_train, X_val)
fi_df, best_iteration1, val_preds = make_val_prediction(X_train, y_train, X_val, y_val, category_cols=category_cols1)
# 如果在真实的情况下，如果train数据集不变的话，那么就可以直接的不要这一步了。把best_iteration 保存下来就行


# 第三部， 进行处理，产出在全量数据上的效果
X, X_test, category_cols = fe1(X, X_test)
preds = make_test_prediction(X, y, X_test, best_iteration1, category_cols=category_cols)

# 所以就是他这里说的，在进行hold out 的时候，要去掉val 的数据上的fraud 标签进行target encoding
# 同时这里的所有特征处理，特征空间都是基于train上的，而不算hold-out 上的特征空间
# 所以要处理多次，时间是两倍的时间，但是make_val  会获得稳定的validation的效果，正相关plb 和prib

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    #'metric': 'None',
    'learning_rate': 0.01,
    'num_leaves': 2**8,
    'max_bin': 255,
    'max_depth': -1,
    'bagging_freq': 5,
    'bagging_fraction': 0.7,
    'bagging_seed': SEED,
    'feature_fraction': 0.7,
    'feature_fraction_seed': SEED,
    'first_metric_only': True,
    'verbose': 100,
    'n_jobs': -1,
    'seed': SEED,
} 

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np

action_1_path = "./data/JData_Action_201602.csv"
action_2_path = "./data/JData_Action_201603.csv"

action_3_path = "./data/JData_Action_201604.csv"
comment_path = "./data/JData_Comment.csv"
product_path = "./data/JData_Product.csv"
user_path = "./data/JData_User.csv"

comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", "2016-03-07", "2016-03-14",
                "2016-03-21", "2016-03-28",
                "2016-04-04", "2016-04-11", "2016-04-15"]


def get_basic_user_feat():
    dump_path = './cache/basic_user.pkl'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path))
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        user['age'] = user['age'].replace({'15岁以下': 0,
                             '16-25岁': 1,
                             '26-35岁': 2,
                             '36-45岁': 3,
                             '46-55岁': 4,
                             '55岁以上': 5,
                             '-1': -1})#map(convert_age)
        user['sex'][user.sex==2.0] = 1.0
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
        user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)
        pickle.dump(user, open(dump_path, 'wb'))
    return user

def get_basic_product_feat():
    dump_path = './cache/basic_product.pkl'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path))
    else:
        product = pd.read_csv(product_path)
        attr1_df = pd.get_dummies(product["a1"], prefix="a1")
        attr2_df = pd.get_dummies(product["a2"], prefix="a2")
        attr3_df = pd.get_dummies(product["a3"], prefix="a3")
        product = pd.concat([product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
        pickle.dump(product, open(dump_path, 'wb'))
    return product


def get_actions_1():
    action = pd.read_csv(action_1_path, encoding='gbk')
    return action


def get_actions_2():
    action2 = pd.read_csv(action_2_path, encoding='gbk')
    return action2


def get_actions_3():
    action3 = pd.read_csv(action_3_path, encoding='gbk')
    return action3


def get_actions(start_date, end_date):
    """

    :param start_date:
    :param end_date:
    :return: actions: pd.Dataframe
    """
    dump_path = './cache/all_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        action_1 = get_actions_1()
        action_2 = get_actions_2()
        action_3 = get_actions_3()
        actions = pd.concat([action_1, action_2, action_3])  # type: pd.DataFrame
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

def get_action_feat(start_date, end_date):#添加
    dump_path = './cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[['user_id', 'sku_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        del actions['type']
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

def get_accumulate_action_feat(start_date, end_date):
    dump_path = './cache/action_accumulate_new_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1) # type: pd.DataFrame
        #近期行为按时间衰减
        actions['weights'] = actions['time'].map(lambda x: datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        #actions['weights'] = time.strptime(end_date, '%Y-%m-%d') - actions['datetime']
        actions['weights'] = actions['weights'].map(lambda x: math.exp(-x.days))
        #print (actions.head(10))
        actions['action_1'] = actions['action_1'] * actions['weights']
        actions['action_2'] = actions['action_2'] * actions['weights']
        actions['action_3'] = actions['action_3'] * actions['weights']
        actions['action_4'] = actions['action_4'] * actions['weights']
        actions['action_5'] = actions['action_5'] * actions['weights']
        actions['action_6'] = actions['action_6'] * actions['weights']
        del actions['model_id']
        del actions['type']
        del actions['time']
        #del actions['datetime']
        del actions['weights']
        actions = actions.groupby(['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


'''
1.用户最近点击、收藏、加购物车、购买时间
2、用户点击、收藏、加购物车、购买量
3、用户转化率即用户购买量分别除以用户点击、收藏、加购物车这三类行为数
4、用户点击、收藏、加购物车、购买量在28天里的均值方差（不按周期计算）
'''


def get_accumulate_user_feat(start_date, end_date):
    feature = ['user_id',
               'user_action_1_sum', 'user_action_2_sum',
               'user_action_3_sum', 'user_action_4_sum',
               'user_action_5_sum', 'user_action_6_sum',
               'user_action_1_ratio', 'user_action_2_ratio', 'user_action_3_ratio',
               'user_action_5_ratio', 'user_action_6_ratio',
               'user_action_mean', 'user_action_std']
    dump_path = './cache/user_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        # actions = pd.concat([actions[['user_id','time']], df], axis=1)
        actions = pd.concat([actions['user_id'], df], axis=1)
        # detime = datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_time, '%Y-%m-%d')
        # actions['weights'] = actions['weights'].map(lambda x: math.exp(-x.days))
        # actions['time'] = pd.to_datetime(actions['time'])
        actions = actions.groupby(['user_id'], as_index=False).sum()  # 以下为各种行为的转化率
        # actions['action_1']=actions['action_1']
        # actions['action_2'] =
        # actions['weekday'] = actions['time'].apply(lambda x: x.weekday() + 1)
        # 转化率
        actions['user_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['user_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['user_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['user_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['user_action_6_ratio'] = actions['action_4'] / actions['action_6']
        # 计数
        actions['user_action_1_sum'] = actions['action_1']
        actions['user_action_2_sum'] = actions['action_2']
        actions['user_action_3_sum'] = actions['action_3']
        actions['user_action_4_sum'] = actions['action_4']
        actions['user_action_5_sum'] = actions['action_5']
        actions['user_action_6_sum'] = actions['action_6']
        # 均值
        # det = datetime.strptime(end_date, '%Y-%m-%d')-datetime.strptime(start_date, '%Y-%m-%d')
        # det = det.days
        actions['user_action_mean'] = actions[['user_action_1_sum',
                                               'user_action_2_sum',
                                               'user_action_3_sum',
                                               'user_action_4_sum',
                                               'user_action_5_sum',
                                               'user_action_6_sum']].mean(axis=1)
        # 方差
        actions['user_action_std'] = actions[['user_action_1_sum', 'user_action_2_sum',
                                              'user_action_3_sum', 'user_action_4_sum',
                                              'user_action_5_sum', 'user_action_6_sum'
                                              ]].std(axis=1)

        actions = actions[feature]
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions

def get_comments_product_feat(start_date, end_date):
    dump_path = './cache/comments_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        comments = pickle.load(open(dump_path))
    else:
        comments = pd.read_csv(comment_path)
        comment_date_end = end_date
        comment_date_begin = comment_date[0]
        for date in reversed(comment_date):
            if date < comment_date_end:
                comment_date_begin = date
                break
        comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]
        df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
        comments = pd.concat([comments, df], axis=1) # type: pd.DataFrame
        #del comments['dt']
        #del comments['comment_num']
        comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]
        pickle.dump(comments, open(dump_path, 'wb'))
    return comments


def get_accumulate_product_feat(start_date, end_date):
    feature = ['sku_id', 'product_action_1_ratio', 'product_action_2_ratio', 'product_action_3_ratio',
               'product_action_5_ratio', 'product_action_6_ratio',
               'product_action_1_sum', 'product_action_2_sum', 'product_action_3_sum',
               'product_action_4_sum', 'product_action_5_sum', 'product_action_6_sum',
               'product_action_mean', 'product_action_std', ]
    dump_path = './cache/product_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['sku_id'], df], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['product_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['product_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['product_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['product_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['product_action_6_ratio'] = actions['action_4'] / actions['action_6']

        actions['product_action_1_sum'] = actions['action_1']
        actions['product_action_2_sum'] = actions['action_2']
        actions['product_action_3_sum'] = actions['action_3']

        actions['product_action_4_sum'] = actions['action_4']
        actions['product_action_5_sum'] = actions['action_5']
        actions['product_action_6_sum'] = actions['action_6']

        # mean,std
        actions['product_action_mean'] = actions[['product_action_1_sum', 'product_action_2_sum',
                                                  'product_action_3_sum', 'product_action_4_sum',
                                                  'product_action_5_sum', 'product_action_6_sum']].mean(axis=1)
        actions['product_action_std'] = actions[['product_action_1_sum', 'product_action_2_sum',
                                                 'product_action_3_sum', 'product_action_4_sum',
                                                 'product_action_5_sum', 'product_action_6_sum']].std(axis=1)

        actions = actions[feature]
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def get_cate_feat(start_date, end_date):
    feature = ['cate', 'cate_action_1_ratio', 'cate_action_2_ratio', 'cate_action_3_ratio',
               'cate_action_5_ratio', 'cate_action_6_ratio',
               'cate_action_1_sum', 'cate_action_2_sum', 'cate_action_3_sum',
               'cate_action_4_sum', 'cate_action_5_sum', 'cate_action_6_sum',
               'cate_action_mean', 'cate_action_std', ]
    dump_path = './cache/cate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['cate'], df], axis=1)
        actions = actions.groupby(['cate'], as_index=False).sum()

        actions['cate_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['cate_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['cate_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['cate_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['cate_action_6_ratio'] = actions['action_4'] / actions['action_6']

        actions['cate_action_1_sum'] = actions['action_1']
        actions['cate_action_2_sum'] = actions['action_2']
        actions['cate_action_3_sum'] = actions['action_3']

        actions['cate_action_4_sum'] = actions['action_4']
        actions['cate_action_5_sum'] = actions['action_5']
        actions['cate_action_6_sum'] = actions['action_6']

        # mean,std
        actions['cate_action_mean'] = actions[['cate_action_1_sum', 'cate_action_2_sum',
                                               'cate_action_3_sum', 'cate_action_4_sum',
                                               'cate_action_5_sum', 'cate_action_6_sum']].mean(axis=1)
        actions['cate_action_std'] = actions[['cate_action_1_sum', 'cate_action_2_sum',
                                              'cate_action_3_sum', 'cate_action_4_sum',
                                              'cate_action_5_sum', 'cate_action_6_sum']].std(axis=1)

        actions = actions[feature]

        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def get_brand_feat(start_date, end_date):
    feature = ['brand', 'brand_action_1_ratio', 'brand_action_2_ratio', 'brand_action_3_ratio',
               'brand_action_5_ratio', 'brand_action_6_ratio',
               'brand_action_1_sum', 'brand_action_2_sum', 'brand_action_3_sum',
               'brand_action_4_sum', 'brand_action_5_sum', 'brand_action_6_sum',
               'brand_action_mean', 'brand_action_std', 'brand_bad_rate']
    dump_path = './cache/brand_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['brand'], df], axis=1)
        actions = actions.groupby(['brand'], as_index=False).sum()

        actions['brand_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['brand_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['brand_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['brand_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['brand_action_6_ratio'] = actions['action_4'] / actions['action_6']

        actions['brand_action_1_sum'] = actions['action_1']
        actions['brand_action_2_sum'] = actions['action_2']
        actions['brand_action_3_sum'] = actions['action_3']

        actions['brand_action_4_sum'] = actions['action_4']
        actions['brand_action_5_sum'] = actions['action_5']
        actions['brand_action_6_sum'] = actions['action_6']

        # mean,std
        actions['brand_action_mean'] = actions[['brand_action_1_sum', 'brand_action_2_sum',
                                                'brand_action_3_sum', 'brand_action_4_sum',
                                                'brand_action_5_sum', 'brand_action_6_sum']].mean(axis=1)
        actions['brand_action_std'] = actions[['brand_action_1_sum', 'brand_action_2_sum',
                                               'brand_action_3_sum', 'brand_action_4_sum',
                                               'brand_action_5_sum', 'brand_action_6_sum']].std(axis=1)

        '''
        原本想添加每个品牌的差评率，但是现在发现有问题，是用分梯度之后的数据计算的，并非实际评论数
        误差多大后面再考虑，暂时先这样
        '''
        rre = pd.read_csv('brand_comment_0201_0415.csv')
        rre = rre.fillna(0)

        actions = pd.merge(actions, rre, how='left', on='brand')
        actions = actions.fillna({'brand_bad_rate': 0})

        actions = actions[feature]

        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def get_accumulate_product_cate_ratio_feat(start_date, end_date):
    '''
    商品各项操作，占该商品所在的大类，各项操作的比例
    '''
    dump_path = './cache/product_cate_ratio_feat_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        # product = pd.read_csv('/Users/ghuihui/jd/data/JData_Product.csv',encoding='gbk')

        # ui = pd.merge(actions,product,how='left',on='sku_id')
        df = pd.get_dummies(actions['type'], prefix='action')
        ui_v = pd.concat([actions, df], axis=1)
        tmp_m = ui_v.groupby(['sku_id', 'cate'], as_index=False).sum()
        tmp_n = tmp_m[['sku_id', 'cate', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6']]
        tmp_t = tmp_n.groupby(tmp_n['sku_id'], as_index=False).sum()
        tmp_tt = tmp_n.groupby(tmp_n['cate'], as_index=False).sum()
        del [tmp_tt['sku_id']]

        tmp_res = pd.merge(tmp_t, tmp_tt, how='left', on='cate')

        tmp_res['action_1_sku_cate_ratio'] = tmp_res['action_1_x'] / tmp_res['action_1_y']
        tmp_res['action_2_sku_cate_ratio'] = tmp_res['action_2_x'] / tmp_res['action_2_y']
        tmp_res['action_3_sku_cate_ratio'] = tmp_res['action_3_x'] / tmp_res['action_3_y']
        tmp_res['action_4_sku_cate_ratio'] = tmp_res['action_4_x'] / tmp_res['action_4_y']
        tmp_res['action_5_sku_cate_ratio'] = tmp_res['action_5_x'] / tmp_res['action_5_y']
        tmp_res['action_6_sku_cate_ratio'] = tmp_res['action_6_x'] / tmp_res['action_6_y']
        fea = ['sku_id',
               'action_1_sku_cate_ratio',
               'action_2_sku_cate_ratio',
               'action_3_sku_cate_ratio',
               'action_4_sku_cate_ratio',
               'action_5_sku_cate_ratio',
               'action_6_sku_cate_ratio']
        tmp_res = tmp_res[fea]
        pickle.dump(tmp_res, open(dump_path, 'wb'))
    return tmp_res


def get_accumulate_product_brand_ratio_feat(start_date, end_date):
    dump_path = './cache/product_brand_ratio_feat_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        # product = pd.read_csv('/Users/ghuihui/jd/data/JData_Product.csv',encoding='gbk')

        # ui = pd.merge(actions,product,how='left',on='sku_id')
        df = pd.get_dummies(actions['type'], prefix='action')
        ui_v = pd.concat([actions, df], axis=1)
        tmp_m = ui_v.groupby(['sku_id', 'brand'], as_index=False).sum()
        tmp_n = tmp_m[['sku_id', 'brand', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6']]
        tmp_t = tmp_n.groupby(tmp_n['sku_id'], as_index=False).sum()
        tmp_tt = tmp_n.groupby(tmp_n['brand'], as_index=False).sum()
        del [tmp_tt['sku_id']]

        tmp_res = pd.merge(tmp_t, tmp_tt, how='left', on='brand')

        tmp_res['action_1_sku_brand_ratio'] = tmp_res['action_1_x'] / tmp_res['action_1_y']
        tmp_res['action_2_sku_brand_ratio'] = tmp_res['action_2_x'] / tmp_res['action_2_y']
        tmp_res['action_3_sku_brand_ratio'] = tmp_res['action_3_x'] / tmp_res['action_3_y']
        tmp_res['action_4_sku_brand_ratio'] = tmp_res['action_4_x'] / tmp_res['action_4_y']
        tmp_res['action_5_sku_brand_ratio'] = tmp_res['action_5_x'] / tmp_res['action_5_y']
        tmp_res['action_6_sku_brand_ratio'] = tmp_res['action_6_x'] / tmp_res['action_6_y']
        fea = ['sku_id',
               'action_1_sku_brand_ratio',
               'action_2_sku_brand_ratio',
               'action_3_sku_brand_ratio',
               'action_4_sku_brand_ratio',
               'action_5_sku_brand_ratio',
               'action_6_sku_brand_ratio']
        tmp_res = tmp_res[fea]
        pickle.dump(tmp_res, open(dump_path, 'wb'))
    return tmp_res

def get_labels(start_date, end_date):
    dump_path = './cache/labels_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 4]
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'sku_id', 'label']]
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, days=30):
    dump_path = './cache/train_set_%s_%s_%s_%s.pkl' % (
    train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        start_days = "2016-02-01"
        user = get_basic_user_feat()
        product = get_basic_product_feat()
        user_acc = get_accumulate_user_feat(train_start_date, train_end_date)
        product_acc = get_accumulate_product_feat(train_start_date, train_end_date)
        comment_acc = get_comments_product_feat(train_start_date, train_end_date)
        cate_acc = get_cate_feat(train_start_date, train_end_date)
        brand_acc = get_brand_feat(train_start_date, train_end_date)

        pro_cate = get_accumulate_product_cate_ratio_feat(train_start_date, train_end_date)
        pro_brand = get_accumulate_product_brand_ratio_feat(train_start_date, train_end_date)
        labels = get_labels(test_start_date, test_end_date)

        # generate 时间窗口
        # actions = get_accumulate_action_feat(train_start_date, train_end_date)
        actions = None
        for i in (1, 2, 3, 5, 7, 10):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_action_feat(start_days, train_end_date)  # get_action_feat(start_days, train_end_date)
            else:
                actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',
                                   on=['user_id', 'sku_id'])

        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, user_acc, how='left', on='user_id')
        actions = pd.merge(actions, product, how='left', on='sku_id')
        actions = pd.merge(actions, product_acc, how='left', on='sku_id')
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')

        actions = pd.merge(actions, pro_cate, how='left', on='sku_id')
        actions = pd.merge(actions, pro_brand, how='left', on='sku_id')

        actions = pd.merge(actions, cate_acc, how='left', on='cate')
        actions = pd.merge(actions, brand_acc, how='left', on='brand')

        actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
        actions = actions.fillna(0)
        pickle.dump(actions, open(dump_path, 'wb'))

    users = actions[['user_id', 'sku_id']].copy()
    labels = actions['label'].copy()
    del actions['user_id']
    del actions['sku_id']
    del actions['label']

    return users, actions, labels


def make_test_set(train_start_date, train_end_date):
    dump_path = './cache/test_set_%s_%s.pkl' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        start_days = "2016-02-01"
        user = get_basic_user_feat()
        product = get_basic_product_feat()
        user_acc = get_accumulate_user_feat(train_start_date, train_end_date)
        product_acc = get_accumulate_product_feat(train_start_date, train_end_date)
        #
        comment_acc = get_comments_product_feat(train_start_date, train_end_date)
        cate_acc = get_cate_feat(train_start_date, train_end_date)
        brand_acc = get_brand_feat(train_start_date, train_end_date)
        pro_cate = get_accumulate_product_cate_ratio_feat(train_start_date, train_end_date)
        pro_brand = get_accumulate_product_brand_ratio_feat(train_start_date, train_end_date)
        # labels = get_labels(test_start_date, test_end_date)

        # generate 时间窗口
        # actions = get_action_feat(train_start_date, train_end_date)
        actions = None
        for i in (1, 2, 3, 5, 7, 10):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_action_feat(start_days, train_end_date)
            else:
                actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',
                                   on=['user_id', 'sku_id'])

        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, user_acc, how='left', on='user_id')
        actions = pd.merge(actions, product, how='left', on='sku_id')
        actions = pd.merge(actions, product_acc, how='left', on='sku_id')
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
        actions = pd.merge(actions, pro_cate, how='left', on='sku_id')
        actions = pd.merge(actions,pro_brand,how='left',on='sku_id')


        actions = pd.merge(actions, cate_acc, how='left', on='cate')
        actions = pd.merge(actions, brand_acc, how='left', on='brand')
        actions = actions.fillna(0)
        actions = actions[(actions['cate'] == 8)]  # |(actions['cate'] == 6)]
        pickle.dump(actions, open(dump_path, 'wb'))

    users = actions[['user_id', 'sku_id']].copy()
    del actions['user_id']
    del actions['sku_id']
    return users, actions

def report(pred, label):

    actions = label
    result = pred

    # 所有用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    # 所有购买用户
    all_user_set = actions['user_id'].unique()

    # 所有品类中预测购买的用户
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)

    # 计算所有用户购买评价指标
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / ( pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print ('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
    print ('所有用户中预测购买用户的召回率' + str(all_user_recall))

    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / ( pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print ('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
    print ('所有用户中预测购买商品的召回率' + str(all_item_recall))
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print ('F11=' + str(F11))
    print ('F12=' + str(F12))
    print ('score=' + str(score))


#from gen_feat import make_train_set
#from gen_feat import make_test_set
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
#from gen_feat import report

def xgboost_make_submission():
    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2,
                                                        random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate': 0.1, 'n_estimators': 1500, 'max_depth': 6,
             'min_child_weight': 1, 'gamma': 2, 'subsample': 0.9, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 380
    param['nthread'] = 4
    # param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    bst = xgb.train(plst, dtrain, num_round, evallist)

    sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date, )
    sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)
    y = bst.predict(sub_trainning_data)
    sub_user_index['label'] = y
    pred = sub_user_index[sub_user_index['label'] >= 0.025]
    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred.to_csv('./sub/submission.csv', index=False, index_label=False)


if __name__ == '__main__':
    #xgboost_cv()
    xgboost_make_submission()


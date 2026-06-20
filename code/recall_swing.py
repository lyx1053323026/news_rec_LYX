import argparse
import math
import os
import pickle
import random
import signal
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('thread')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='swing 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'swing 召回，mode: {mode}')

# Swing 算法参数
MAX_USERS_PER_ITEM = 300  # 每个物品最多保留的用户数，超出则随机采样
MAX_ITEMS_PER_USER = 200  # 每个用户最多保留的历史物品数


def cal_sim(df):
    # 限制每个用户的历史物品数（保留最近的）
    df = df.sort_values(['user_id', 'click_timestamp'])
    df = df.groupby('user_id').tail(MAX_ITEMS_PER_USER)

    # 构建用户-物品和物品-用户映射
    user_item_ = df.groupby('user_id')['click_article_id'].agg(list).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    item_user_ = df.groupby('click_article_id')['user_id'].agg(list).reset_index()
    item_user_dict = dict(
        zip(item_user_['click_article_id'], item_user_['user_id']))

    log.info(f'过滤后: {len(user_item_dict)} 用户, {len(item_user_dict)} 物品')

    # 将用户物品列表转为 set 加速交集计算
    user_item_set = {u: set(items) for u, items in user_item_dict.items()}

    sim_dict = {}

    for item, users in tqdm(item_user_dict.items()):
        sim_dict.setdefault(item, {})

        # 对热门物品采样用户，控制计算量
        if len(users) > MAX_USERS_PER_ITEM:
            users = random.sample(users, MAX_USERS_PER_ITEM)

        user_list = sorted(users)
        for a in range(len(user_list)):
            u = user_list[a]
            u_items = user_item_set[u]
            for b in range(a + 1, len(user_list)):
                v = user_list[b]
                v_items = user_item_set[v]

                shared_items = u_items & v_items
                overlap = len(shared_items)
                weight = 1.0 / (1.0 + overlap)

                for relate_item in shared_items:
                    if relate_item != item:
                        sim_dict[item].setdefault(relate_item, 0)
                        sim_dict[item][relate_item] += weight

    return sim_dict, user_item_dict


@multitasking.task
def recall(df_query, item_sim, user_item_dict, worker_id):
    data_list = []

    for user_id, item_id in tqdm(df_query.values):
        rank = {}

        if user_id not in user_item_dict:
            continue

        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:2]

        for loc, item in enumerate(interacted_items):
            if item not in item_sim:
                continue
            for relate_item, wij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:100]:
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij * (0.7**loc)

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:100]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('../user_data/tmp/swing', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/swing/{worker_id}.pkl')


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/sim/offline', exist_ok=True)
        sim_pkl_file = '../user_data/sim/offline/swing_sim.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/sim/online', exist_ok=True)
        sim_pkl_file = '../user_data/sim/online/swing_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    item_sim, user_item_dict = cal_sim(df_click)
    f = open(sim_pkl_file, 'wb')
    pickle.dump(item_sim, f)
    f.close()

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('../user_data/tmp/swing'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, item_sim, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('../user_data/tmp/swing'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = pd.concat([df_data, df_temp], sort=False)

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'swing: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_swing.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_swing.pkl')

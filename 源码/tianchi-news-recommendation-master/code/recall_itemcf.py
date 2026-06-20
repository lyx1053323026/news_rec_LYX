import argparse
import math
import os
import pickle
import random
import signal
from collections import defaultdict
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
parser = argparse.ArgumentParser(description='itemcf 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'itemcf 召回，mode: {mode}')


def make_item_time_tuple(group_df):
    """将分组后的DataFrame转换为(item, click_timestamp)元组列表"""
    return list(zip(group_df['click_article_id'], group_df['click_timestamp']))


def get_user_item_time(df):
    """
    根据点击时间获取用户的点击文章序列（按时间排序）
    v2策略：保留时间戳信息用于后续权重计算
    return: dict, {user1: [(item1, time1), (item2, time2)..]...}
    """
    df = df[['user_id', 'click_article_id', 'click_timestamp']]
    df = df.sort_values('click_timestamp')
    user_item_time_df = df.groupby('user_id')[['click_article_id', 'click_timestamp']]\
        .apply(lambda group: make_item_time_tuple(group))\
        .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    return user_item_time_dict


def get_item_info_dict(article_df):
    """
    获取文章id对应的基本属性（v2策略）
    return:
        item_created_abs_time_dict: {article_id: raw_created_at_ts} 绝对时间戳（毫秒）
        item_created_time_dict: {article_id: normalized_created_at_ts} min-max归一化时间
    """
    # 获取绝对时间字典
    item_created_abs_time_dict = dict(zip(article_df['article_id'], article_df['created_at_ts']))

    # 获取归一化时间字典 (min-max scaling)
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    article_df['created_at_ts_norm'] = article_df[['created_at_ts']].apply(max_min_scaler)
    item_created_time_dict = dict(zip(article_df['article_id'], article_df['created_at_ts_norm']))

    return item_created_abs_time_dict, item_created_time_dict


def cal_sim(df, item_created_time_dict):
    """
    文章与文章之间的相似性矩阵计算（v2策略）
    改进点：
    - 考虑文章的正向顺序点击和反向顺序点击：正向权重1.0，反向权重0.7
    - 考虑文章的位置信息权重：0.8**(|loc2-loc1|-1)，距离越远权重越低
    - 考虑文章的点击时间权重：exp(0.8**|Δt_click|)，时间越近权重越高
    - 考虑文章创建时间的权重：exp(0.8**|Δt_created|)，创建时间越近权重越高
    - 对热门物品进行非对称惩罚：wij/(max_cnt^0.4 * min_cnt^0.6)

    return: i2i_sim, user_item_time_dict
    """
    user_item_time_dict = get_user_item_time(df)

    # 计算相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)

    for user, item_time_list in tqdm(user_item_time_dict.items()):
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if i == j:
                    continue

                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 位置信息权重，使用0.8衰减因子（v2策略）
                loc_weight = loc_alpha * (0.8 ** (np.abs(loc2 - loc1) - 1))
                # 点击时间权重：同一时间戳点击的权重更高
                click_time_weight = np.exp(0.8 ** np.abs(i_click_time - j_click_time))
                # 两篇文章创建时间的权重：创建时间相近的权重更高
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))

                i2i_sim[i].setdefault(j, 0)
                # 计算文章之间的相似度，用户序列长度作为归一化因子
                i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(len(item_time_list) + 1)

    # 两篇文章的流行度权重，惩罚过于热门的物品（v2策略：非对称惩罚）
    i2i_sim_ = i2i_sim.copy()
    popular_weight = 0.4
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            tmpMax, tmpMin = max(item_cnt[i], item_cnt[j]), min(item_cnt[i], item_cnt[j])
            i2i_sim_[i][j] = wij / ((tmpMax ** popular_weight) * (tmpMin ** (1 - popular_weight)))

    return i2i_sim_, user_item_time_dict


@multitasking.task
def recall(df_query, item_sim, user_item_time_dict, item_created_time_dict,
           item_created_abs_time_dict, hot_items, worker_id):
    """
    基于文章协同过滤的召回（v2策略）
    改进点：
    - 文章创建时间窗口过滤：只保留创建时间在[last_click-1.8e8, last_click+1e5]内的候选
    - 文章创建时间差权重：exp(0.9**|Δt_created|)，创建时间越接近权重越高
    - 历史点击位置权重：0.9^(len-loc)，越靠近最后一次点击权重越高
    - 不足recall_item_num时用热门物品补全
    """
    data_list = []
    sim_item_topk = 200
    recall_item_num = 100

    # 统计热门补齐情况
    total_users = 0
    users_need_padding = 0
    total_padded_items = 0

    for user_id, item_id in tqdm(df_query.values):
        rank = {}

        if user_id not in user_item_time_dict:
            continue

        total_users += 1
        user_hist_items = user_item_time_dict[user_id]
        user_hist_items_set = {item for item, _ in user_hist_items}
        last_click_time = user_hist_items[-1][1]

        for loc, (i, click_time) in enumerate(user_hist_items):
            if i not in item_sim:
                continue

            for j, wij in sorted(item_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
                if j in user_hist_items_set:
                    continue

                # 文章创建时间窗口过滤：不在合理时间区间的直接pass（v2强特）
                if j in item_created_abs_time_dict:
                    if (item_created_abs_time_dict[j] > last_click_time + 1 * (10 ** 5) or
                            item_created_abs_time_dict[j] < last_click_time - 1.8 * (10 ** 8)):
                        continue

                # 文章创建时间差权重
                if i in item_created_time_dict and j in item_created_time_dict:
                    created_time_weight = np.exp(0.9 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                else:
                    created_time_weight = 1.0

                # 相似文章和历史点击文章序列中历史文章所在的位置权重
                # 离最后一次点击越近的文章，其相似文章的权重越高
                loc_weight = (0.9 ** (len(user_hist_items) - loc))

                rank.setdefault(j, 0)
                rank[j] += created_time_weight * loc_weight * wij

        # 不足recall_item_num个，用热门商品补全
        normal_recall_cnt = len(rank)
        if len(rank) < recall_item_num:
            users_need_padding += 1
            padded_cnt = 0
            for k, hot_item in enumerate(hot_items):
                if hot_item in rank:
                    continue
                rank[hot_item] = - k - 100  # 给一个负数分数，确保排在正常召回结果之后
                padded_cnt += 1
                if len(rank) == recall_item_num:
                    break
            total_padded_items += padded_cnt

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:recall_item_num]
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

    os.makedirs('../user_data/tmp/itemcf', exist_ok=True)
    df_data.to_pickle(f'../user_data/tmp/itemcf/{worker_id}.pkl')

    # 保存补齐统计
    stats = {
        'total_users': total_users,
        'users_need_padding': users_need_padding,
        'total_padded_items': total_padded_items,
    }
    pickle.dump(stats, open(f'../user_data/tmp/itemcf/{worker_id}_stats.pkl', 'wb'))


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/sim/offline', exist_ok=True)
        sim_pkl_file = '../user_data/sim/offline/itemcf_sim.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/sim/online', exist_ok=True)
        sim_pkl_file = '../user_data/sim/online/itemcf_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    # 加载文章信息（v2新增：用于获取文章创建时间）
    article_df = pd.read_csv('../../../tcdata/articles.csv')
    item_created_abs_time_dict, item_created_time_dict = get_item_info_dict(article_df)
    log.debug(f'文章数量: {len(item_created_abs_time_dict)}')

    # 计算热门文章列表（v2新增：用于召回不足时补充）
    hot_items = df_click['click_article_id'].value_counts().index.tolist()
    log.debug(f'热门文章数量: {len(hot_items)}')

    # 计算相似度矩阵（v2策略），如果已缓存则直接加载
    if os.path.exists(sim_pkl_file):
        log.info('加载已缓存的相似度矩阵...')
        item_sim = pickle.load(open(sim_pkl_file, 'rb'))
        user_item_time_dict = get_user_item_time(df_click)
    else:
        item_sim, user_item_time_dict = cal_sim(df_click, item_created_time_dict)
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
    for path, _, file_list in os.walk('../user_data/tmp/itemcf'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, item_sim, user_item_time_dict, item_created_time_dict,
               item_created_abs_time_dict, hot_items, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    total_users = 0
    users_need_padding = 0
    total_padded_items = 0

    for path, _, file_list in os.walk('../user_data/tmp/itemcf'):
        for file_name in file_list:
            if file_name.endswith('_stats.pkl'):
                stats = pickle.load(open(os.path.join(path, file_name), 'rb'))
                total_users += stats['total_users']
                users_need_padding += stats['users_need_padding']
                total_padded_items += stats['total_padded_items']
                continue
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = pd.concat([df_data, df_temp], sort=False)

    log.info(f'热门补齐统计: 总用户数={total_users}, 需要补齐的用户数={users_need_padding}, '
             f'补齐比例={users_need_padding/total_users*100:.2f}%, '
             f'总共补齐文章数={total_padded_items}, '
             f'平均每用户补齐数={total_padded_items/total_users:.2f}')

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True, False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_itemcf.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_itemcf.pkl')

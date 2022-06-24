import pickle

import implicit
import numpy as np
import pandas as pd


def load_data_and_model():
    df = pd.read_csv('users_history.csv')
    gb = df.groupby(['user_id', 'article_id']).size().to_frame().reset_index()
    gb.rename(columns = {0:'cliks'}, inplace = True)
    unique_users = gb.user_id.unique()
    users_ids = dict(zip(unique_users, np.arange(unique_users.shape[0], dtype=np.int32)))
    unique_items = gb.article_id.unique()
    item_ids = dict(zip(unique_items, np.arange(unique_items.shape[0], dtype=np.int32)))
    gb['u_id'] = gb.user_id.apply(lambda i: users_ids[i])
    gb['a_id'] = gb.article_id.apply(lambda i: item_ids[i])
    model = pickle.load(open('collab_model.p', 'rb'))
    return df, gb, model

def collaborative_filtering(user_id, top_n, df, gb, model):
    user_story = df[df.user_id == user_id] #filter for user clicks story
    if user_story.empty:
        return f'No reads history for {user_id}'
    article_id = user_story[user_story.click_timestamp == max(user_story.click_timestamp)].article_id.values[0] #find last clicked article
    a_id = gb[gb.article_id == article_id].a_id.values[0]
    related = model.similar_items(a_id, N=top_n+1)
    #delete already read if any
    if np.any(related[0] == a_id):
        reco = np.delete(related[0], np.where(related[0] == a_id))
    else:
        reco = related[0][:-1].copy()
    articles = gb[gb.a_id.isin(reco)].article_id.unique()
    return articles.tolist()

import json
import logging

import azure.functions as func
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

articles_embeddings = pd.read_pickle('articles_embeddings.pickle')
df = pd.read_csv('users_history.csv')
df['click_timestamp'] = pd.to_datetime(df.click_timestamp)

def get_n_similar_articles(article_id, top_n):
    sim_matrix = cosine_similarity(articles_embeddings, articles_embeddings[article_id].reshape(1, -1))
    idx = np.argsort(sim_matrix, axis=0)[::-1][1:top_n+1]
    return idx.flatten().tolist()

def content_filtering(user_id, df, top_n=5):
    user_story = df[df.user_id == int(user_id)]
    # last clicked article
    article_id = user_story[user_story.click_timestamp == max(user_story.click_timestamp)].article_id.values[0]
    recommend = get_n_similar_articles(article_id, top_n)
    return recommend

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    user_id = req.params.get('userId')
    if not user_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            user_id = req_body.get('userId')

    if user_id:
        logging.info(user_id)
        rec = content_filtering(user_id, df)
        return func.HttpResponse(json.dumps(rec))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully, but did no understand the userId.",
             status_code=200
        )

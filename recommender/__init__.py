import json
import logging

import azure.functions as func
from collaborative_filtering import (collaborative_filtering,
                                     load_data_and_model)


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
        logging.info(f'Generating recommendations for {user_id}...')
        df, gb, model = load_data_and_model()
        rec = collaborative_filtering(int(user_id), 5, df, gb, model)
        return func.HttpResponse(json.dumps(rec))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully, but did no understand the userId.",
             status_code=200
        )

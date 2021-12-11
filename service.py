# -*- coding: utf-8 -*-
import os
import numpy as np
from typing import Any

from flask import Flask, request, abort

import prediction_model


def init_model() -> Any:
    return prediction_model.PredictionModel()


model = init_model()


def create_data_for_model(current_request: dict) -> np.array:
    data: str = current_request['data']
    categories_name = []
    arr = []
    for category in data:
        cat_id: str = category["cat_id"]
        categories_name.append(cat_id)
        sold: list[float] = category["sold"]
        arr.append(sold)
    return np.array(arr).transpose(), categories_name


def create_result_money_json(arr: np.array, user_id: str, cat_id: list[str]) -> dict:
    result = {"user_id": str(user_id)}
    data = []
    arr = arr.transpose()
    for idx, category_predict in enumerate(arr):
        cat = {"cat_id": cat_id[idx], "predict_four_weeks": list(map(float, arr[idx])), "predict_month": sum(arr[idx])}
        data.append(cat)
    result["data"] = data
    return result


def create_result_item_json(arr: np.array, user_id: str, cat_id: list[str]) -> dict:
    result = {"user_id": str(user_id)}
    data = []
    arr = arr.transpose()
    for idx, category_predict in enumerate(arr):
        cat = {"cat_id": cat_id[idx], "predict_four_weeks": list(map(int, arr[idx])), "predict_month": int(sum(arr[idx]))}
        data.append(cat)
    result["data"] = data
    return result


def create_app() -> Flask:
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    @app.route('/index')
    def index() -> Any:
        return '''<p>Что вершит судьбу человечества в этом мире?</p>
        <p>Некое незримое существо или закон, подобно Длани Господней, парящей над миром?</p>
        <p>По крайне мере истинно то, что человек не властен даже над своей волей.</p>'''

    @app.route('/predict_spent_money', methods=['GET'])
    def predict_spent_money() -> Any:
        try:
            if not model:
                abort(503)

            current_request: dict = request.json
            user_id: str = current_request['user_id']
            data_for_model, categories_name = create_data_for_model(current_request)
            result = model.predict_next_month_sum_distribution(data_for_model)
            return create_result_money_json(result, user_id, categories_name)

        except TypeError or KeyError:
            abort(400)

    @app.route('/predict_purchased_goods', methods=['GET'])
    def predict_purchased_goods() -> Any:
        try:
            if not model:
                abort(503)

            current_request: dict = request.json
            user_id: str = current_request['user_id']
            data_for_model, categories_name = create_data_for_model(current_request)
            result = model.predict_next_month_sum_distribution(data_for_model)
            return create_result_item_json(result, user_id, categories_name)

        except TypeError or KeyError:
            abort(400)

    return app


if __name__ == '__main__':
    app_ = create_app()
    app_.run(port=8095)

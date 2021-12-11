from typing import Callable

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from lstm_model import LstmModel

INPUT_SIZE = 29
HIDDEN_SIZE = 42
NUM_LAYERS = 1
OUTPUT_SIZE = 29
LINEAR_LAYER_SIZE = 30
SERIES_SIZE = 6
MONTH_PERIOD = 4


def predict(predict_model: Callable[[torch.tensor], torch.tensor], week_distribution: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler(feature_range=(0, 1))
    prep_input = scaler.fit_transform(week_distribution)
    tensor_input = torch.tensor(prep_input.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        output = predict_model(tensor_input)
    return scaler.inverse_transform(output.numpy())


def predict_on_period(predict_model: Callable[[torch.tensor], torch.tensor], week_distribution: np.ndarray,
                      series_size: int = SERIES_SIZE, period_len: int = MONTH_PERIOD) -> np.ndarray:
    model_input_buffer = np.zeros(shape=(week_distribution.shape[0] + period_len, week_distribution.shape[1]))
    model_input_buffer[0:series_size] = week_distribution
    for i in range(0, period_len):
        output = predict(predict_model, model_input_buffer[i:i + series_size - 1]).squeeze()
        model_input_buffer[i + series_size] = output
    prediction = model_input_buffer[-period_len::]
    return prediction


class PredictionModel:
    def __init__(self, sum_model_path: str = 'model/model_distribution_sums.pth',
                 item_model_path: str = 'model/model_distribution_items.pth'):
        self.sum_prediction_model = LstmModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                                              output_size=OUTPUT_SIZE, linear_layer_size=LINEAR_LAYER_SIZE)
        self.sum_prediction_model.load_state_dict(torch.load(sum_model_path))
        self.sum_prediction_model.eval()

        self.item_prediction_model = LstmModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                                               output_size=OUTPUT_SIZE, linear_layer_size=LINEAR_LAYER_SIZE)
        self.item_prediction_model.load_state_dict(torch.load(item_model_path))
        self.item_prediction_model.eval()

    def predict_next_week_sum_distribution(self, week_sum_distribution: np.ndarray) -> np.ndarray:
        return predict(predict_model=self.sum_prediction_model, week_distribution=week_sum_distribution).squeeze()

    def predict_next_week_item_distribution(self, week_item_distribution: np.ndarray) -> np.ndarray:
        return np.floor(
            predict(predict_model=self.item_prediction_model, week_distribution=week_item_distribution)).squeeze()

    def predict_next_month_sum_distribution(self, week_sum_distribution: np.ndarray) -> np.ndarray:
        return predict_on_period(predict_model=self.sum_prediction_model, week_distribution=week_sum_distribution)

    def predict_next_month_item_distribution(self, week_item_distribution: np.ndarray) -> np.ndarray:
        return predict_on_period(predict_model=self.item_prediction_model,
                                 week_distribution=week_item_distribution)

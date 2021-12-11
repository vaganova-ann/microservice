# service.py - файл с микросервисом для модели.  
  
Пример входных данных: request.txt  
Пример выходных данных:  
  для предсказания денег: result_money.txt  
  для предсказания продуктов: result_item.txt  

Получить предсказание количества потраченных денег по категориям на следующий месяц: http://127.0.0.1:8095/predict_spent_money  
Получить предсказание количества купленых товаров по категориям: http://127.0.0.1:8095/predict_purchased_goods  
Оба метода являются get-методами.  

---

# hmc_production_model - это итоговая модель для микросервиса предсказаний по курсу КРПО,проект "home money control"

lstm_model.py содержит класс с архитектурой нейронной сети(используется только в prediction_model.py)

prediction_model.py содержит класс **PredictionModel**,который и необходимо **использовать для предсказания при помощи
методов описанных ниже**

example.py  содержит пример работы с PredictionModel

в requirements.txt лежит информация о всех необходимых библиотеках для использвания 


### Описание методов класса PredictionModel



```
predict_next_week_sum_distribution(week_sum_distribution: np.ndarray) -> np.ndarray
```

**вход**: week_sum_distribution: np.ndarray размерности 6 x 29, каждая строка которого содержит распределение
сумм,потраченных на каждую из категорий

**выход**: np.array размерности 29 - распределение сумм, потраченных на каждую категорию в предсказанную неделю

```
predict_next_week_item_distribution(week_item_distribution: np.ndarray) -> np.ndarray
```

**вход**: week_item_distribution: np.ndarray размерности 6 x 29, каждая строка которого содержит распределение количества
предметов,купленных в каждой из категорий

**выход**: np.ndarray размерности 29 - распределение количества предметов, купленных в каждой категории в предсказанную
неделю

```
predict_next_month_sum_distribution(self, week_sum_distribution: np.ndarray) -> np.ndarray:
```

**вход**: week_sum_distribution: np.ndarray размерности 6 x 29, каждая строка которого содержит распределение
сумм,потраченных на каждую из категорий

**выход**: np.ndarray размерности 4 * 29 - распределение сумм, потраченных на каждую категорию в каждую неделю месяца ( в
месяце 4 недели)

```
predict_next_month_item_distribution(self, week_item_distribution: np.ndarray) -> np.ndarray:
```
**вход**:week_item_distribution: np.ndarray размерности 6 x 29, каждая строка которого содержит распределение количества
предметов,купленных в каждой из категорий

**выход**: np.array размерности 4 * 29 - распределение предметов , купленных в  каждойкатегории в каждую неделю месяца ( в
месяце 4 недели)


### Создание объекта класса PredictionModel и пример использования

```
  PredictionModel( sum_model_path: str = 'model/model_distribution_sums.pth',
                 item_model_path: str = 'model/model_distribution_items.pth'):
```

для инициализации объекта необходоимо передать в конструктор путь до  сохраненных данных моделей для предсказания сумм и объектов.
при использовании аргументов по умолчанию  будут использованы  пути до моделей, лежащих в папке model в репозитории

**вход**: sum_model_path - путь до модели для предсказания распределения сумм .pth,str

item_model_path - путь до модели для предсказания распределения предметов .pth, str

```python
import numpy as np
import prediction_model


model = prediction_model.PredictionModel()
sample = np.array([[1] * 29 for _ in range(6)])
print(model.predict_next_month_sum_distribution(sample))
```

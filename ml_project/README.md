<h3 align="center">ml_project</h3>
Для подготовки данных к обучению необходимо запустить файл prepare.py.
В качестве флагов к нему можно передать путь до данных(--data-path), название файла с данными(--data-name),
долю данных для обучения(--train-size)
По умолчанию эти значения будут взяты из файла configs/random_forest_config.json

Для обучения модели необходимо запустить файл train.py.
В качестве флагов к нему можно передать тип модели(--model-type), путь для сохранения модели(--model-path),
название файла с моделью(--model-name), названия колонок с категориальными данными(--categorical-cols),
параметры для RandomForest (--n-estimator, --n-jobs)
По умолчанию эти значения будут взяты из файла configs/random_forest_config.json

Для получения данных из модели необходимо запустить файл predict.py.
В качестве флагов к нему можно передать путь для сохранения модели(--model-path), название файла с моделью(--model-name),
путь для сохранения решения(--solution-path), название файла с решением(--solution-name).
По умолчанию эти значения будут взяты из файла configs/random_forest_config.json


Пример использования:

<$ python prepare.py --train-size 0.75 >

$ python train.py --model-type "LinearRegression" --n-estimator 150 --categorical-cols "sex cp fbs" 

$ python predict.py --solution-path "." --solution-name "solution"


Запуск тестов:

$ python tests.py


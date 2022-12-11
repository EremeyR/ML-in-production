<h3 align="center">online inference</h3>

Создание docker image производится командой из корневой директории:

$ docker build -t inference:0.5 -f ./online_inference/Dockerfile .

Запуск docker контейнера:

$ docker run -p 8100:8100 inference:0.5

Запуск getter-файла:

$ cd online_inference

$ python getter.py

Запуск тестов:

$ cd online_inference

$ pytest

<h3 align="center">airflow_ml_dags</h3>

### Для корректной работы с переменными, созданными из UI
export FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker compose up --build

Перед использованием 03_predict необходимо определить variable "current_model_path", в которой указать путь до модели относительно папки data
(airflow -> Admin -> Variables) 
Пример: Key:current_model_path, Val:

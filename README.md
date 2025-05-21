[![Video Demo](https://img.youtube.com/vi/Oah91n-3b2A/0.jpg)](https://youtu.be/Oah91n-3b2A)

# Project Structure
```
simpsons/
├── .dvc/                  # DVC configs
├── conf/                  # Hydra configs
├── data/                  # Data
│   └── journey-springfield/
├── docs/                  # Documentation
├── models/                # Model check-points via DVC
├── plots/                 # Loss and metrics visualization
├── simpsons/              # The main package
│   ├── __init__.py
│   ├── augmentation.py    # Image augmentation
│   ├── classifier.py      # Model architecture
│   ├── convert_to_onnx.py # ONNX conversion
│   ├── dataset.py         # Data preprocessing
│   ├── inference.py       # Prediction logic
│   ├── model.py           # PyTorch Lightning module
│   └── train.py           # Training pipeline
├── tests/                 # Unit tests
├── triton/                # Triton Inference Server configs
├── wandb/                 # Weights & Biases logs
├── .pre-commit-config.yaml # Git hooks
├── pyproject.toml         # Poetry config
├── poetry.lock            # Dependency locks
└── README.md              # Project documentation
```

# Simpsons Character Classifier
## Setup
```powershell
# Установка Poetry
# pip
pip install poetry==2.1.2
# powershell
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -

# Клонирование репозитория
git clone https://github.com/RenataKostolina/simpsons
cd simpsons

# Установка зависимостей
poetry install

# Инициализация DVC
poetry run dvc init

# Установка хуков
poetry run pre-commit install

# Запуск проверок
poetry run pre-commit run --all-files

```
## Testing
```powershell
# Запуск тестов
poetry run pytest tests/ -v
```

## Train
```powershell
# Активация окружения (если не активировано)
poetry shell

# Авторизация в Weights & Biases
poetry run wandb login  # Введите API-ключ при запросе

# Загрузка данных через DVC (должны загрузиться самостоятельно при запуске следующего шага, но если этого не произошло, то нужно воспользоваться командой)
poetry run dvc pull data/journey-springfield

# Тренировка модели
poetry run python ./simpsons/train.py

# После обучения в директории сonf/inference/ измените поле ckpt на путь к вашей модели или добавьте исходную с помощью
poetry run dvc pull models/
```

## Production preparation
```powershell
# Экспорт в ONNX
poetry run python ./simpsons/convert_to_onnx.py
```
Нужно переместить файл model.onnx в соответствии со следующей архитектурой:
```
triton
|   docker-compose.yml
|   Dockerfile
\---sources
    |   .gitignore
    |   label_encoder.pkl
    |   model.onnx
    |   requirements.txt
    |   triton.py
    \---static
        |   index.html  
        \---images
```
А также установить Docker с официального сайта: https://www.docker.com/products/docker-desktop/

```bash
# Переход в директорию с Triton (убедитесь, что вы находитесь в корне проекта перед выполнением этой команды)
cd triton

# Проверка работающих контейнеров
docker ps

# Сборка образа веб-сервиса
docker-compose build --no-cache web --progress=plain

# Сборка всех сервисов
docker-compose build --no-cache

# Запуск сервисов
docker-compose up
```

Сервис будет доступен в браузере по адресу: http://127.0.0.1:8080/

```bash
# Остановка сервисов
docker-compose down
```

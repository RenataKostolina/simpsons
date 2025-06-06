# Аннотация

## Постановка задачи

В данном проекте мы занимаемся созданием и обучением классификатора на основе
свёрточных нейронных сетей (CNN), который будет способен распознавать и
классифицировать персонажей из популярного мультсериала "Симпсоны".

Цель — разработать модель, способную идентифицировать различных жителей
Спрингфилда на изображениях, что позволит пользователям взаимодействовать с
контентом мультсериала.

### Зачем это нужно?

Образовательные цели: Проект предоставляет возможность изучить и применить
методы глубокого обучения и компьютерного зрения. Работа с реальными данными и
задачами помогает лучше понять принципы работы свёрточных нейронных сетей.

Развлечение: Создание классификатора для персонажей "Симпсонов" может стать
основой для различных развлекательных приложений, таких как игры или викторины,
где пользователи могут тестировать свои знания о персонажах и их
характеристиках.

Потенциал: Успешная реализация классификатора может открыть двери для дальнейших
исследований и разработок в области распознавания лиц и объектов, а также в
других мультимедийных приложениях, таких как анализ изображений и видео.

Таким образом, наша задача не только развивает навыки в области машинного
обучения, но и создает интересный и полезный продукт.

## Формат входных и выходных данных

Обучающая и тестовая выборки состоят из отрывков из мультсериала «Симпсоны».
Каждая картинка представлена в формате .jpg с необходимой меткой — названием
персонажа, изображённого на ней. Метки классов представлены в виде названий
папок, в которых находятся изображения. В обучающем наборе данных примерно по
1000 изображений на каждый класс, но они отличаются по размеру.

Наиболее важным аспектом является то, что этот набор данных включает множество
разнообразных и хорошо размеченных изображений. Это означает, что у нас есть
достаточное количество изображений для каждого персонажа, что позволит нашей
модели обучиться более качественно.

## Метрики

Одной из наиболее релевантных метрик является F1-score.

### Почему F1-score?

F1-score является гармоническим средним между precision и полнотой recall. Это
особенно важно в задачах классификации, где классы могут быть
несбалансированными. В нашем случае, если один из персонажей встречается
значительно реже, чем другие, то высокая точность для более распространенных
классов может скрывать низкую производительность для менее распространённых.
F1-score позволяет учитывать оба аспекта.

В нашей задаче мы имеем дело с большим количеством персонажей, что делает
F1-score удобным для многоклассовой классификации.

## Источник данных

https://www.kaggle.com/competitions/journey-springfield/data

## Моделирование

### Подготовка данных

- Аугментация данных: на самом деле, как окажется, классы в этом датасете не
  сбалансированы, поэтому следует применить аугментацию. В данном случае будем
  применять RandomCrop, RandomRotation,RandomHorizontalFlip, RandomPerspective.
- Предобработка изображений: изображения персонажей будут преобразованы в
  фиксированный размер 224x224, преобразованы в numpy array, нормализованы
  (значения пикселей приведены к диапазону [0, 1]), приведены к torch-тензорам и
  снова нормированы с помощью вычисления среднего и дисперсии.

### Архитектура нейронной сети

Наша нейронная сеть будет иметь 6 сверточных слоев и 2 полносвязных слоя. Каждый
слой со сверткой имеет следующую архитектуру:

```
Сверточный слой,
Активация (ReLU),
Пуллинг (MaxPool),
Нормализация батча,
Исключение (Dropout) - только на первых 4 слоях.
```

Каждый полносвязный слой имеет следующую архитектуру:

```
Полносвязный слой,
Активация (ReLU).
```

# Демонстрация

Click to watch the demo

[![Simpsons Classifier Demo](https://img.youtube.com/vi/Oah91n-3b2A/hqdefault.jpg)](https://youtu.be/Oah91n-3b2A)

# Структура проекта

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

```
# powershell
# Клонирование репозитория
git clone https://github.com/RenataKostolina/simpsons
cd simpsons

# Установка Poetry
# pip
pip install poetry==2.1.2
# powershell
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -

# Установка зависимостей
poetry install

# Установка хуков
poetry run pre-commit install

# Запуск проверок
poetry run pre-commit run --all-files

```

## Testing

```
# powershell
# Запуск тестов
poetry run pytest tests/ -m "not requires_files"

```

## Train

```
# powershell
# Активация окружения (если не активировано)
poetry env activate

# Авторизация в Weights & Biases
poetry run wandb login --relogin  # Введите API-ключ при запросе

# Тренировка модели
poetry run python ./simpsons/train.py
```

## Production preparation

```
# powershell
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

А также установить Docker с официального сайта:
https://www.docker.com/products/docker-desktop/

```
# bash
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

```
# bash
# Остановка сервисов
docker-compose down
```

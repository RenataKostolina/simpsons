import os
import pickle

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

app = FastAPI()

# Монтирование статических файлов
app.mount("/static", StaticFiles(directory="/triton/sources/static"), name="static")

# Загрузка модели и label_encoder
ort_session = ort.InferenceSession("/triton/sources/model.onnx")
with open("/triton/sources/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


# Главная страница
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("/triton/sources/static/index.html")


# API для предсказаний
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Создаем папку для изображений, если ее нет
        os.makedirs("/triton/sources/static/images", exist_ok=True)

        # Сохранение файла
        file_path = f"/triton/sources/static/images/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Обработка изображения
        img = Image.open(file_path).resize((224, 224))

        # Конвертация в numpy array и нормализация
        img_array = np.array(img, dtype=np.float32)

        img_array = img_array / 255.0  # Нормализация [0, 1]
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW

        # Нормализация (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_array = (img_array - mean) / std

        # Добавляем batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Получаем имя входного узла модели
        input_name = ort_session.get_inputs()[0].name

        # Предсказание
        outputs = ort_session.run(None, {input_name: img_array})
        outputs = outputs[0]  # Берем первый выход

        predicted_idx = np.argmax(outputs)
        predicted_label = label_encoder.classes_[predicted_idx]

        return {"class": predicted_label, "image_url": f"/static/images/{file.filename}"}

    except Exception as e:
        # Удаляем файл в случае ошибки
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


def main():
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()

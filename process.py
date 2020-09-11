# Скрипт v.0.0.1 никогда ранее скрпиты не писал поэтому скорее всего он получился корявым)
# Импорт необходимых библиотек
from sys import argv
import os
import sys
# sys.path.insert(1, "<D:\Python learning\new_env\Lib\site-packages\tensorflow>")
import tensorflow as tf
import numpy as np
import json

# Присвоим путь к изображениям
path = str(argv[1])

# Присвоим пути модели и пути сохранения файла json
model_directory = r'C:\Users\User\Jupiter\Gender_detetion\internship_data\models\08-09-2020_14-52-49_10000_img.h5'
path_json = r'C:\Users\User\Jupiter\Gender_detetion\internship_data\test'


files = os.listdir(path)
images = list(filter(lambda x: x.endswith('.jpg'), files))
images_path = [path + '\\' + fname for fname in images]

# Определим размер изображения
IMG_height = 200
IMG_width = 200

# Создадим функцию для препроцессинга изображений


def process_image(image_path, img_height=IMG_height, img_width=IMG_width):
    """
    На вход подается путь к изображению и функция конвертирует изображение в тензор.
    """
    # Считаем изображение
    image = tf.io.read_file(image_path)
    # Конвертируем изображение в формате jpeg в числовой тензор с 3 каналами (красный, зеленый и синий)
    image = tf.image.decode_jpeg(image, channels=3)
    # Конвертируем каналы цвета из диапазано 0-255 в диапазон 0-1(нормализуем)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Изменим размер наших рисунков до (300, 300)
    image = tf.image.resize(image, size=(IMG_height, IMG_width))

    return image


# Определим размер батча, 32
BATCH_SIZE = 32


# Создадим функцию для преобразования в батчи(пакеты)
def create_data_batches(X, batch_size=BATCH_SIZE):
    """
    Создадает пакеты данных для тестовой выборки (X)
    """
    # Если выборка представлет их себя тестовый датасет у нас скорее  всего не будет меток для ее
    print("Создаются пакеты для тестовой выборки...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))  # Только пути к файлам (без меток)
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch


# Создадим батч из тестовой выборки
test_data = create_data_batches(images_path)

# Определим путь к модели


# Загрузим модель
model = tf.keras.models.load_model(model_directory)

# Спрогнозируем классы
predictions = model.predict(test_data, verbose=1)

unique_gender = ['female', 'male']


# Конвертируем уровни достоверности в соответствующую метку

def get_pred_label(prediction_probabilities):
    """
    Конвертирует массив вероятностей в разметку
    """

    return unique_gender[np.argmax(prediction_probabilities)]


keys = images
values = [get_pred_label(predictions[i]) for i in range(len(predictions))]
d = dict(zip(keys, values))

def writeToJSONFile(path, fileName, data):
    filePathNameWExt = path + '\\' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)




writeToJSONFile(path_json, 'process_results', d)

fname = 'process_results.json'
path_json_file = path_json + '\\' + fname
os.path.isfile(path_json_file)
if os.path.isfile(path_json_file) == False:
    print("Ошибка. Файл не сохранился")
else:
    print("Файл process_result.json успешно сохранен")

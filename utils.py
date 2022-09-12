# Модуль функций
import numpy as np
import cv2 as cv
import os
import shutil
import time
import json


# Функция проверки пар картинка/geojson
def files_pair_check(dataset_path):
    img_count = 0
    error_list = []
    all_files = sorted(os.listdir(dataset_path))
    for file in all_files:
        _, file_extension = os.path.splitext(file)
        if file_extension == '.png':
            img_count += 1
            json_name = file.replace("png", "geojson")
            if json_name not in all_files:
                # print('Не нашли файл geojson для изображения {}'.format(file))
                error_list.append(file)
    return img_count, error_list


# Функция сбора информации о классах разметки сегментации
def ann_check(dataset_path, verbose=False):
    # Создадим списки файлов
    img_names = []
    ann_names = []
    for f in sorted(os.listdir(dataset_path)):
        _, file_extension = os.path.splitext(f)
        if file_extension == '.png':
            img_names.append(f)
        elif file_extension == '.geojson':
            ann_names.append(f)
    # print("Нашли изображений {}, аннотаций {}".format(len(img_names), len(ann_names)))

    ann_set = np.array([])
    cur_time = time.time()
    for file_name in img_names:
        if verbose:
            print('В обработке (подсчет классов) (): {}'.format(file_name))
        img_path = os.path.join(dataset_path, file_name)
        json_path = img_path.replace("png", "geojson")
        img = cv.imread(img_path)  # потребуется узнать размер изображения
        ann_channels = get_mask_from_json(json_path, img.shape[:2])  # передадим размер для ч/б картинки

        # Нас будет интересовать только 1 канал (собственно маска )
        # ann = np.zeros(ann_channels.shape[:2], dtype=np.float32)
        # ann = np.where(ann_channels[:, :, 1] >= 1, 1, 0).astype(np.uint8)

        curr_set = np.unique(ann_channels[:,:,1])
        ann_set = np.union1d(ann_set, curr_set)
    print("Время выполнения: ", round(time.time() - cur_time, 2), 'c', sep='')
    return ann_set


# Функция получения маски из json (из baseline)
def get_mask_from_json(path, image_size):
    def parse_polygon(coordinates: dict, image_size: tuple) -> np.ndarray:
        mask = np.zeros(image_size, dtype=np.float32)
        if len(coordinates) == 1:
            points = [np.int32(coordinates)]
            cv.fillPoly(mask, points, 1)
        else:
            for polygon in coordinates:
                points = [np.int32([polygon])]
                cv.fillPoly(mask, points, 1)
        return mask

    def parse_mask(shape, image_size):
        mask = np.zeros(image_size, dtype=np.float32)
        coordinates = shape['coordinates']
        if shape['type'] == 'MultiPolygon':
            for polygon in coordinates:
                mask += parse_polygon(polygon, image_size)
        else:
            mask += parse_polygon(coordinates, image_size)
        return mask

    class_ids = {"vessel": 1}

    with open(path, 'r', encoding='cp1251') as f:
        json_contents = json.load(f)

        num_channels = 1 + max(class_ids.values())  # 2
        mask_channels = [np.zeros(image_size, dtype=np.float32) for _ in range(num_channels)]
        mask = np.zeros(image_size, dtype=np.float32)

        if (type(json_contents) == type({})) and (json_contents['type'] == 'FeatureCollection'):
            features = json_contents['features']
        elif type(json_contents) == list:
            features = json_contents
        else:
            features = [json_contents]

        for shape in features:
            channel_id = class_ids["vessel"]  # 1
            mask = parse_mask(shape['geometry'], image_size)
            mask_channels[channel_id] = np.maximum(mask_channels[channel_id], mask)

        mask_channels[0] = 1 - np.max(mask_channels[1:], axis=0)
        result = np.stack(mask_channels, axis=-1)
        return result


# Функция подготовки данных
def dataset_prep(dataset_path, imgs_path, masks_path, verbose=False):
    # Создадим списки файлов
    img_names = []
    ann_names = []
    for f in sorted(os.listdir(dataset_path)):
        _, file_extension = os.path.splitext(f)
        if file_extension == '.png':
            img_names.append(f)
        elif file_extension == '.geojson':
            ann_names.append(f)
    # print("Нашли изображений {}, аннотаций {}".format(len(img_names), len(ann_names)))
    cur_time = time.time()
    for file_name in img_names:
        # if verbose:
        #     print('В обработке (подготовка данных): {}'.format(file_name))
        img_path = os.path.join(dataset_path, file_name)
        json_path = img_path.replace("png", "geojson")
        img = cv.imread(img_path)  # потребуется узнать размер изображения
        ann_channels = get_mask_from_json(json_path, img.shape[:2])  # передадим размер для ч/б картинки

        # Нас будет интересовать только 1 канал (собственно маска )
        ann = np.zeros(ann_channels.shape[:2], dtype=np.float32)
        ann = np.where(ann_channels[:, :, 1] >= 1, 255, 0).astype(np.uint8)
        # print(img.shape, ann.shape)

        # Скопируем картинку в другую папку
        shutil.copy(os.path.join(dataset_path, file_name), imgs_path)
        if verbose:
            print('Скопировали файл: {}'.format(file_name))
        # Сохраняем маску под именем картинки, но в другую папку
        out_file = os.path.join(masks_path, file_name)
        try:
            cv.imwrite(out_file, ann)
        except IOError:
            print('Не удалось сохранить файл: {}'.format(out_file))
        finally:
            if verbose:
                print('Записали файл: {}'.format(out_file))

    print("Время выполнения: ", round(time.time() - cur_time, 2), 'c', sep='')


# Функция автокоррекции контраста
def autocontrast(img):
    # converting to LAB color space
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    result = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    return result


# Функция преобразования аннотации в one hot encoding
def mask_to_ohe(ann_image, classes=[0, 255]):
    ones = np.ones((ann_image.shape[0], ann_image.shape[1], len(classes)), dtype=np.uint8)
    zeros = np.zeros((ann_image.shape[0], ann_image.shape[1], len(classes)), dtype=np.uint8)

    result = zeros.copy()

    result[:, :, 0] = np.where(ann_image == 0, ones[:, :, 0], zeros[:, :, 0])
    result[:, :, 1] = np.where(ann_image == 255, ones[:, :, 1], zeros[:, :, 1])

    return result


# Функция преобразования аннотации из ohe в классы
def ohe_to_mask(ann_ohe, classes=[0, 255]):
    ones = np.ones((ann_ohe.shape[0], ann_ohe.shape[1]), dtype=np.uint8)
    zeros = np.zeros((ann_ohe.shape[0], ann_ohe.shape[1]), dtype=np.uint8)

    result = zeros.copy()

    result = np.where(ann_ohe[:, :, 0] == 1, ones * 0, result)
    result = np.where(ann_ohe[:, :, 1] == 1, ones * 255, result)

    return result


# Функция ресайза картинки через opencv
def img_resize_cv(image, img_size=1024):
    """
    :param image: исходное изображение
    :param img_size: размер к которому приводить изображение
    :return: изображение после ресайза
    """
    curr_w = image.shape[1]
    curr_h = image.shape[0]
    # Рассчитаем коэффициент для изменения размера
    if curr_w > curr_h:
        scale_img = img_size / curr_w
    else:
        scale_img = img_size / curr_h
    # Новые размеры изображения
    new_width = int(curr_w * scale_img)
    new_height = int(curr_h * scale_img)
    # делаем ресайз к целевым размерам
    image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    return image

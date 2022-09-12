# Подготовка данных для модели сегментации
import os
import utils as u
# Выводить дополнительную информацию
VERBOSE = True

# Папки для картинок
dataset_path = 'train_dataset_mc'
imgs_path = 'imgs'
masks_path = 'masks'

if __name__ == '__main__':
    # Создадим папки для готовых файлов, если их нет
    if not (imgs_path in os.listdir('.')):
        os.mkdir(imgs_path)
    if not (masks_path in os.listdir('.')):
        os.mkdir(masks_path)

    # Проверим датасет на парность картинка/geojson
    img_count, error_list = u.files_pair_check(dataset_path)
    print('Нашли изображений: {}'.format(img_count))
    if len(error_list) > 0:
        print('Не нашли geojson для {} файлов: {}'.format(len(error_list), error_list))
    else:
        print('Для всех изображений нашли файл разметки')

    # Переименуем файлы с пробелом в имени и удалим изображения без разметки
    for file in error_list:
        file_name, file_extension = os.path.splitext(file)
        if ' ' in file_name:
            new_file = file.replace(' ', '')
            # print(os.path.join(dataset_path, file), os.path.join(dataset_path, new_file))
            os.rename(os.path.join(dataset_path, file), os.path.join(dataset_path, new_file))
        else:
            os.remove(os.path.join(dataset_path, file))

    # Если были ошибки, то проверим датасет на парность картинка/geojson ЕЩЕ РАЗ
    if len(error_list) > 0:
        img_count, error_list = u.files_pair_check(dataset_path)
        print('Нашли изображений: {}'.format(img_count))
        if len(error_list) > 0:
            print('Не нашли geojson для {} файлов: {}'.format(len(error_list), error_list))
        else:
            print('Для всех изображений нашли файл разметки')

    # Теперь ошибок быть не должно
    assert len(error_list) == 0

    # Соберем информацию о классах разметки сегментации
    # ann_set = u.ann_check(dataset_path, VERBOSE)
    # print("Получили классы разметки: {}".format(ann_set))

    # Перезаписываем картинки и преобразуем geojson в маску
    u.dataset_prep(dataset_path, imgs_path, masks_path, VERBOSE)




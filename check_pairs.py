# Проверка корректности датасета в части наличия разметки у всех изображений
import os

# Папки для картинок
imgs_path = 'imgs'
masks_path = 'masks'


def files_pair_check(imgs_path = 'imgs', masks_path = 'masks'):
    error_list = []
    imgs_files = sorted(os.listdir(imgs_path))
    ann_files = sorted(os.listdir(masks_path))
    for file in imgs_files:
        if file not in ann_files:
            print('Не нашли файл аннотации для изображения {}'.format(file))
            error_list.append(file)
    return error_list


if __name__ == '__main__':
  # Проверим на парность датасет
  print(files_pair_check())


# Capillaries-segm

Всероссийский чемпионат Цифровой прорыв - 2022, г.Владивосток

Сегментация капилляров глаза человека по снимкам с офтальмологической щелевой лампы.

1. Датасет обработан и подготовлен для загрузки в colab (12-09-22)
2. Обучил первую унетку. Точность визуально весьма слабая. На лидербоарде ....
(потерял время формируя сабмит с фейковыми файлами чтобы было как sample_solution)
3. Гипотеза 1: попробовать унетку с elu вместо relu
4. Гипотеза 2: Сделать функцию метрики как в base_line (f1)
5. Гипотеза 3: учить на фрагментах 512х512 от оригинальных фото
6. Гипотеза 4: попробовать сегментацию по цветовому фильтру

Подготовка материалов к сдаче ?

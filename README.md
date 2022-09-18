# Capillaries-segm

Всероссийский чемпионат Цифровой прорыв - 2022, г.Владивосток

Сегментация капилляров глаза человека по снимкам с офтальмологической щелевой лампы.

1. Датасет обработан и подготовлен для загрузки в colab (12-09-22)
2. Обучил первую унетку. Точность слабая. Подготовил предикт теста и выложил на ЛБ. Потерял время разбираясь "чтобы было как sample_solution" (13-09-22)
3. Попробовал унетку с loss функцией dice, (выучить не получилось, не дают GPU) (13-09-22)
4. Попробовал несколько кастомных loss функций, улучшения точности не достиг (13-09-22)
5. Почистил данные и собрал малый датасет из более надежных по разметке картинок. Результаты теста хуже чем на полном датасете (14-09-22)
6. Сделал датагенератор на фрагментах и учил на нем, результат обучения лучше, но предикт на полноразмерной картинке хуже (15-09-22)
7. Проверил разрешение 384х384 (остальные параметры теже)  - за 50 эпох результат нулевой (15-09-22)
8. Вставил метрику F1 (15-09-22)
9. Проверил обучение если сделать все активационные функции 'elu' - за 50 эпох результат почти нулевой (15-09-22)
10. Идея с "крошечным" датасетом отпала (будет падение точности на тесте, аналогично "малому" датасету) (16-09-22)
11. Предварительно сделал CV-модель с фильтрацией по цвету (где нейронка дает только набор цветов) (16-09-22)
12. Доделал CV-модель (фильтр по уникальным цветам из маски). Результат ЛБ практически не изменился (17-09-22)
13. Сделал и обучил модель с использованием слоев от VGG16. Обучение не плучилось (18-09-22)
14. 
15. 


Подготовка материалов к сдаче: презентация, тизер и т.д. ... ?

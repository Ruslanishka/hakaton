import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Загрузка датасета
df = pd.read_csv('df_hack_final.csv')

# Датапрепарация
df = df.dropna()  # Удаление строк с пропущенными значениями

# Все признаки
features = [
    'Cu_1.1C', 'Cu_1.2C', 'Cu_2.1C', 'Cu_2.2C', 'Cu_2F', 'Cu_3.1C', 'Cu_3.2C',
    'Cu_3F', 'Cu_4F', 'Cu_oreth', 'Cu_resth', 'Dens_1', 'Dens_2', 'Dens_3',
    'Dens_4', 'Dens_5', 'Dens_6', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4',
    'Mass_5', 'Mass_6', 'Ni_1.1C', 'Ni_1.2C', 'Ni_2.1C', 'Ni_2.2C', 'Ni_2F',
    'Ni_3.1C', 'Ni_3.2C', 'Ni_3F', 'Ni_4.1C', 'Ni_4.2C', 'Ni_4F', 'Ni_5.1C',
    'Ni_5.2C', 'Ni_5F', 'Ni_6.1C', 'Ni_6.2C', 'Ni_6F', 'Ni_oreth', 'Ni_resth',
    'Ore_mass', 'Vol_4', 'Vol_5', 'Vol_6'
]

# Группировка признаков по итерациям
iterations = {}
for i in range(1, 7):
    iterations[i] = [col for col in features if f"_{i}" in col or f"{i}C" in col or f"{i}F" in col]

# Проверка наличия признаков в датасете
iterations_with_data = {}
for i, cols in iterations.items():
    available_cols = [col for col in cols if col in df.columns]
    iterations_with_data[i] = available_cols

# Целевые переменные (диапазоны)
targets = {
    1: ['Ni_1.1C_max', 'Ni_1.1C_min', 'Ni_1.1T_max', 'Ni_1.1T_min'],
    2: ['Ni_2.1C_max', 'Ni_2.1C_min', 'Ni_2.1T_max', 'Ni_2.1T_min'],
    3: ['Ni_3.1C_max', 'Ni_3.1C_min', 'Ni_3.1T_max', 'Ni_3.1T_min'],
    4: ['Ni_4.1C_max', 'Ni_4.1C_min', 'Ni_4.1T_max', 'Ni_4.1T_min'],
    5: ['Ni_5.1C_max', 'Ni_5.1C_min', 'Ni_5.1T_max', 'Ni_5.1T_min'],
    6: ['Ni_6.1C_max', 'Ni_6.1C_min', 'Ni_6.1T_max', 'Ni_6.1T_min']
}

def generate_ranges(predictions):
    ranges = []
    for pred in predictions:
        generated = []
        for i in range(0, len(pred), 2):
            min_value = np.round(pred[i + 1], 1)  
            max_value = np.round(pred[i], 1)  
            
            if min_value >= max_value:
                min_value = max_value - 0.1
            
            generated.extend([min_value, max_value])
        
        ranges.append(generated)
    
    return np.array(ranges)

# Создание пустого DataFrame для хранения результатов
all_results = pd.DataFrame()

# Обучение моделей для каждой итерации
for i in range(1, 7):
    print(f"Итерация {i}:")

    # Признаки и целевые переменные текущей итерации
    X_cols = iterations_with_data[i]
    y_cols = targets[i]

    # Проверка наличия целевых переменных
    y_cols = [col for col in y_cols if col in df.columns]

    if not X_cols or not y_cols:
        print(f"Пропущена итерация {i}: нет доступных данных.")
        continue

    # Признаки и целевые переменные
    X = df[X_cols]
    y = df[y_cols]

    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Обучение нейронной сети
    from keras.layers import Input

    model = Sequential()
    model.add(Input(shape=(X_scaled.shape[1],)))  # Используйте слой Input для определения формы входных данных
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y.shape[1]))  # Выходной слой
    model.compile(loss='mse', optimizer='adam')

    # Используем EarlyStopping для предотвращения переобучения
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Обучение модели
    model.fit(X_scaled, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Предсказание
    y_pred = model.predict(X_scaled)

    # Генерация диапазонов
    predicted_ranges = generate_ranges(y_pred)

    # Сохранение результатов для текущей итерации в общий DataFrame
    iteration_results = pd.DataFrame(predicted_ranges, columns=y_cols)
    iteration_results['Iteration'] = i  # Добавление колонки с номером итерации
    all_results = pd.concat([all_results, iteration_results], ignore_index=True)

# Сохранение всех результатов в один файл
all_results.to_csv('test_results/all_iterations_results.csv', index=False)
print("Все результаты сохранены в test_results/all_iterations_results.csv")
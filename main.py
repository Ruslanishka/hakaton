import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping

# Загрузка датасета
df = pd.read_csv('df_hack_final.csv')

# Заполнение строк (медиана)
df.iloc[:, 1:] = df.iloc[:, 1:].fillna(df.iloc[:, 1:].median())

# Создание списка признаков, включая все столбцы
features = df.columns.tolist()

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
    1: ['Ni_1.1C_min', 'Ni_1.1C_max', 'Ni_1.1T_min', 'Ni_1.1T_max', 'Cu_1.1C_min', 'Cu_1.1C_max'],
    2: ['Ni_1.2C_min', 'Ni_1.2C_max', 'Cu_1.2C_min', 'Cu_1.2C_max', 'Cu_2.1T_min', 'Cu_2.1T_max'],
    3: ['Cu_2.2T_min', 'Cu_2.2T_max', 'Cu_3.1T_min', 'Cu_3.1T_max', 'Cu_3.2T_min', 'Cu_3.2T_max'],
    4: ['Ni_4.1C_min', 'Ni_4.1C_max', 'Ni_4.1T_min', 'Ni_4.1T_max', 'Ni_4.2C_min', 'Ni_4.2C_max', 'Ni_4.2T_min', 'Ni_4.2T_max'],
    5: ['Ni_5.1C_min', 'Ni_5.1C_max', 'Ni_5.1T_min', 'Ni_5.1T_max', 'Ni_5.2C_min', 'Ni_5.2C_max', 'Ni_5.2T_min', 'Ni_5.2T_max'],
    6: ['Ni_6.1C_min', 'Ni_6.1C_max', 'Ni_6.1T_min', 'Ni_6.1T_max', 'Ni_6.2C_min', 'Ni_6.2C_max', 'Ni_6.2T_min', 'Ni_6.2T_max']
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

    # Обучение ней ронной сети
    model = Sequential()
    model.add(Input(shape=(X_scaled.shape[1],)))
    model.add(Dense(128, activation='relu'))  # Увеличение количества нейронов
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y.shape[1]))  # Выходной слой
    model.compile(loss='mse', optimizer='adam')

    # Используем EarlyStopping для предотвращения переобучения
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Обучение модели
    model.fit(X_scaled, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Предсказание
    y_pred = model.predict(X_scaled)

    # Проверка на наличие NaN в предсказаниях
    if np.any(np.isnan(y_pred)):
        print(f"Итерация {i}: Предсказания содержат NaN.")
        continue  # Пропустить итерацию, если есть NaN

    # Генерация диапазонов
    predicted_ranges = generate_ranges(y_pred)

    # Проверка на пропущенные значения в сгенерированных диапазонах
    if np.any(np.isnan(predicted_ranges)):
        print(f"Итерация {i}: Сгенерированные диапазоны содержат NaN.")
        continue  # Пропустить итерацию, если есть NaN

    # Сохранение результатов для текущей итерации в общий DataFrame
    iteration_results = pd.DataFrame(predicted_ranges, columns=y_cols)
    iteration_results['Iteration'] = i  # Добавление колонки с номером итерации
    iteration_results['MEAS_DT'] = df['MEAS_DT'].iloc[:len(predicted_ranges)].values  # Добавление столбца MEAS_DT
    cols = ['MEAS_DT'] + [col for col in iteration_results.columns if col != 'MEAS_DT']
    
    all_results = pd.concat([all_results, iteration_results], ignore_index=True)

# Сохранение всех результатов в один файл
all_results.to_csv('test_results/test.csv', index=False)
print("Все результаты сохранены в test_results/test.csv")
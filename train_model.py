import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# --- 1. CARGA Y LIMPIEZA DE DATOS ---
print("Cargando dataset...")
df = pd.read_csv('data.csv', sep='\t')

# Filtrar edades lógicas
df = df[(df['age'] >= 13) & (df['age'] <= 100)]

# Columnas TIPI
tipi_cols = [f'TIPI{i}' for i in range(1, 11)]
for col in tipi_cols:
    df = df[df[col] > 0]

# --- 2. INGENIERÍA DE VARIABLES (DASS COMPLETO) ---

# Ítems de Depresión, Ansiedad y Estrés según DASS-42
dep_items = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
anx_items = [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]
str_items = [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]

dep_cols = [f'Q{i}A' for i in dep_items]
anx_cols = [f'Q{i}A' for i in anx_items]
str_cols = [f'Q{i}A' for i in str_items]

# Calculamos los puntajes (restamos 1 por ítem para pasar 1–4 -> 0–3)
df['DepressionScore'] = df[dep_cols].sum(axis=1) - len(dep_cols)
df['AnxietyScore']    = df[anx_cols].sum(axis=1) - len(anx_cols)
df['StressScore']     = df[str_cols].sum(axis=1) - len(str_cols)

# Función de categorización en tres niveles
def categorize(score):
    if score <= 13:
        return 0   # Bajo / Normal
    elif score <= 20:
        return 1   # Medio / Moderado
    else:
        return 2   # Alto / Severo

df['Dep_Level'] = df['DepressionScore'].apply(categorize)
df['Anx_Level'] = df['AnxietyScore'].apply(categorize)
df['Str_Level'] = df['StressScore'].apply(categorize)

# --- 3. VARIABLES DE ENTRADA (X) Y SALIDA (Y) ---

# Entrada: 10 TIPI + Edad
X = df[tipi_cols + ['age']].values  # 11 características

# Salidas crudas (0,1,2) para cada escala
y_dep = df['Dep_Level'].values
y_anx = df['Anx_Level'].values
y_str = df['Str_Level'].values

print(f"Muestras disponibles: {X.shape[0]}")

# --- 4. PREPROCESAMIENTO ---

X_train, X_test, y_dep_train, y_dep_test, y_anx_train, y_anx_test, y_str_train, y_str_test = train_test_split(
    X, y_dep, y_anx, y_str,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# One-hot encoding para cada salida
y_dep_train_cat = to_categorical(y_dep_train, 3)
y_dep_test_cat  = to_categorical(y_dep_test, 3)

y_anx_train_cat = to_categorical(y_anx_train, 3)
y_anx_test_cat  = to_categorical(y_anx_test, 3)

y_str_train_cat = to_categorical(y_str_train, 3)
y_str_test_cat  = to_categorical(y_str_test, 3)

# --- 5. MODELO MLP MULTISALIDA ---

input_layer = Input(shape=(X_train_scaled.shape[1],), name='input')

x = Dense(32, activation='relu')(input_layer)
x = Dropout(0.3)(x)
x = Dense(16, activation='relu')(x)

# Tres salidas, una por cada subescala
dep_out = Dense(3, activation='softmax', name='dep_out')(x)
anx_out = Dense(3, activation='softmax', name='anx_out')(x)
str_out = Dense(3, activation='softmax', name='str_out')(x)

model = Model(inputs=input_layer, outputs=[dep_out, anx_out, str_out])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'dep_out': 'categorical_crossentropy',
        'anx_out': 'categorical_crossentropy',
        'str_out': 'categorical_crossentropy'
    },
    metrics={
        'dep_out': 'accuracy',
        'anx_out': 'accuracy',
        'str_out': 'accuracy'
    }
)


print("Entrenando red neuronal (Depresión, Ansiedad, Estrés)...")

history = model.fit(
    X_train_scaled,
    [y_dep_train_cat, y_anx_train_cat, y_str_train_cat],
    epochs=30,
    batch_size=32,
    verbose=1,
    validation_split=0.1
)

# Evaluación
eval_results = model.evaluate(X_test_scaled, [y_dep_test_cat, y_anx_test_cat, y_str_test_cat], verbose=0)

# Keras devuelve: [loss_total, dep_loss, dep_acc, anx_loss, anx_acc, str_loss, str_acc]
print(f"Accuracy Depresión (test): {eval_results[2]*100:.2f}%")
print(f"Accuracy Ansiedad  (test): {eval_results[4]*100:.2f}%")
print(f"Accuracy Estrés    (test): {eval_results[6]*100:.2f}%")

# --- 6. GUARDAR MODELO Y ESCALADOR ---

model.save('model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Modelo multi-salida guardado. Sistema listo.")

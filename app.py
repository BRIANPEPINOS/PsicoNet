from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)

# Cargar modelo y escalador
model = tf.keras.models.load_model('model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def get_recommendations(dep_level, anx_level, str_level):
    """Genera recomendaciones basadas en los niveles de riesgo"""
    recs = []
    
    if anx_level == "Riesgo Alto (Significativo)":
        recs.append(("Técnica 4-7-8", "Inhala 4s, mantén 7s, exhala 8s. Ayuda a calmar el sistema nervioso."))
        recs.append(("Mindfulness", "Dedica 5 minutos a observar objetos a tu alrededor sin juzgarlos."))
    
    if str_level == "Riesgo Alto (Significativo)":
        recs.append(("Método Pomodoro", "Trabaja 25 min y descansa 5. Reduce la carga mental."))
        recs.append(("Desconexión Digital", "Evita pantallas 1 hora antes de dormir."))
        
    if dep_level == "Riesgo Alto (Significativo)":
        recs.append(("Activación Conductual", "Haz una actividad pequeña que antes disfrutabas, aunque no tengas ganas."))
        recs.append(("Diario de Gratitud", "Escribe 3 cosas simples por las que estás agradecido hoy."))

    if not recs:
        recs.append(("Mantenimiento", "Tus niveles parecen estables. Sigue practicando tus hábitos actuales."))
        
    return recs

@app.route('/', methods=['GET', 'POST'])
def index():
    context = {} # Diccionario para pasar variables al HTML

    if request.method == 'POST':
        try:
            # 1. Recolectar datos
            features = []
            # TIPI 1-10
            for i in range(1, 11):
                val = float(request.form[f'tipi{i}'])
                features.append(val)
            
            age = float(request.form['age'])
            features.append(age)

            # 2. Preprocesar y Predecir
            input_data = np.array([features])
            input_scaled = scaler.transform(input_data)
            pred_dep, pred_anx, pred_str = model.predict(input_scaled)

            # 3. Decodificar etiquetas
            labels = ["Riesgo Bajo (Normal)", "Riesgo Medio (Moderado)", "Riesgo Alto (Significativo)"]
            colors = ["success", "warning", "danger"]

            dep_idx = np.argmax(pred_dep[0])
            anx_idx = np.argmax(pred_anx[0])
            str_idx = np.argmax(pred_str[0])

            # 4. CALCULAR BIG FIVE (TIPI SCORING)
            # Indices en 'features': TIPI1 es features[0], etc.
            # Escala inversa: (8 - valor)
            
            # Extraversión: TIPI1, TIPI6(R)
            extraversion = (features[0] + (8 - features[5])) / 2
            
            # Amabilidad (Agreeableness): TIPI2(R), TIPI7
            agreeableness = ((8 - features[1]) + features[6]) / 2
            
            # Responsabilidad (Conscientiousness): TIPI3, TIPI8(R)
            conscientiousness = (features[2] + (8 - features[7])) / 2
            
            # Estabilidad Emocional: TIPI4(R), TIPI9
            emotional_stability = ((8 - features[3]) + features[8]) / 2
            
            # Apertura (Openness): TIPI5, TIPI10(R)
            openness = (features[4] + (8 - features[9])) / 2

            # Guardar todo en el contexto
            context = {
                'show_results': True,
                'dep_prediction': labels[dep_idx],
                'anx_prediction': labels[anx_idx],
                'str_prediction': labels[str_idx],
                'dep_class': colors[dep_idx],
                'anx_class': colors[anx_idx],
                'str_class': colors[str_idx],
                # Datos para el gráfico
                'radar_data': [extraversion, agreeableness, conscientiousness, emotional_stability, openness],
                # Recomendaciones
                'recommendations': get_recommendations(labels[dep_idx], labels[anx_idx], labels[str_idx])
            }

        except Exception as e:
            context['error'] = f"Error: {str(e)}"

    return render_template('index.html', **context)

if __name__ == '__main__':
    app.run(debug=True)
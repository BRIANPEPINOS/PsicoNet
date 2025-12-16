from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import pickle
import tflite_runtime.interpreter as tflite
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash
from datetime import datetime
import os

app = Flask(__name__)

# --- CONFIGURACIÓN ---
app.config['SECRET_KEY'] = 'clave_secreta_super_segura'
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'neurodata.db')

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- MODELOS BD ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password_hash = db.Column(db.String(200))
    name = db.Column(db.String(100))
    role = db.Column(db.String(20)) 

class SurveyResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    date = db.Column(db.DateTime, default=datetime.now)
    dep_result = db.Column(db.String(50))
    anx_result = db.Column(db.String(50))
    str_result = db.Column(db.String(50))
    user = db.relationship('User', backref=db.backref('results', lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- CARGA DEL MODELO ---
model_path = os.path.join(basedir, 'model.tflite')
scaler_path = os.path.join(basedir, 'scaler.pkl')

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# --- RUTAS ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard') if user.role == 'admin' else url_for('index'))
        else:
            flash('Usuario o contraseña incorrectos')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if current_user.role == 'admin': return redirect(url_for('dashboard'))
    context = {}

    if request.method == 'POST':
        try:
            # 1. Datos
            features = [float(request.form[f'tipi{i}']) for i in range(1, 11)]
            features.append(float(request.form['age']))
            
            # 2. Preprocesar
            input_scaled = scaler.transform(np.array([features])).astype(np.float32)
            
            # 3. PREDICCIÓN TFLITE
            interpreter.set_tensor(input_details[0]['index'], input_scaled)
            interpreter.invoke()
            
            pred_dep = interpreter.get_tensor(output_details[0]['index'])
            pred_anx = interpreter.get_tensor(output_details[1]['index'])
            pred_str = interpreter.get_tensor(output_details[2]['index'])
            
            labels = ["Riesgo Bajo", "Riesgo Medio", "Riesgo Alto"]
            colors = ["success", "warning", "danger"]
            
            d_idx = np.argmax(pred_dep[0])
            a_idx = np.argmax(pred_anx[0])
            s_idx = np.argmax(pred_str[0])
            
            # 4. Guardar
            new_result = SurveyResult(
                user_id=current_user.id,
                dep_result=labels[d_idx],
                anx_result=labels[a_idx],
                str_result=labels[s_idx]
            )
            db.session.add(new_result)
            db.session.commit()

            # 5. Visuales
            radar_data = [
                (features[0] + (8-features[5]))/2, ((8-features[1]) + features[6])/2,
                (features[2] + (8-features[7]))/2, ((8-features[3]) + features[8])/2,
                (features[4] + (8-features[9]))/2
            ]
            
            recs = []
            if "Alto" in labels[a_idx]: recs.append(("Respiración 4-7-8", "Inhala 4s, sostén 7s, exhala 8s."))
            if "Alto" in labels[d_idx]: recs.append(("Activación", "Haz una actividad pequeña."))
            if not recs: recs.append(("Mantenimiento", "Todo parece estable."))

            context = {
                'show_results': True,
                'dep_prediction': labels[d_idx], 'dep_class': colors[d_idx],
                'anx_prediction': labels[a_idx], 'anx_class': colors[a_idx],
                'str_prediction': labels[s_idx], 'str_class': colors[s_idx],
                'radar_data': radar_data, 'recommendations': recs
            }
            
        except Exception as e:
            flash(f"Error técnico: {str(e)}")

    return render_template('index.html', **context, name=current_user.name)

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role != 'admin': return "Acceso denegado", 403
    
    all_results = SurveyResult.query.order_by(SurveyResult.date.desc()).all()
    
    # --- CORRECCIÓN AQUÍ: Agregamos 'str' al diccionario ---
    stats = {'dep': {}, 'anx': {}, 'str': {}}
    
    for r in all_results:
        stats['dep'][r.dep_result] = stats['dep'].get(r.dep_result, 0) + 1
        stats['anx'][r.anx_result] = stats['anx'].get(r.anx_result, 0) + 1
        stats['str'][r.str_result] = stats['str'].get(r.str_result, 0) + 1 # Contamos estrés
        
    return render_template('dashboard.html', results=all_results, stats=stats)
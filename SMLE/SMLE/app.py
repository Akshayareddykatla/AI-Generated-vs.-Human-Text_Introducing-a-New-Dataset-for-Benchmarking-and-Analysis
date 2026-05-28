from flask import Flask, render_template, request, redirect, url_for, session, flash
from models import db, User, DetectionReport, SMLEModel
import json
import plotly.express as px
import plotly.utils

app = Flask(__name__)
app.secret_key = 'smle_secure_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db.init_app(app)
model = SMLEModel()

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username'], password=request.form['password']).first()
        if user:
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        flash('Invalid credentials.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = User(username=request.form['username'], password=request.form['password'])
        db.session.add(user); db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session: return redirect(url_for('login'))
    data = None
    if request.method == 'POST':
        text = request.form.get('text_content')
        if text:
            res = model.predict(text)
            shap_data = model.get_shap_values(text)
            report = DetectionReport(
                input_text=text, 
                label_results=json.dumps(res), 
                shap_scores=json.dumps(shap_data), 
                user_id=session['user_id']
            )
            db.session.add(report); db.session.commit()
            data = {"labels": res, "highlights": shap_data}
    return render_template('dashboard.html', data=data)

@app.route('/reports')
def reports():
    if 'user_id' not in session: return redirect(url_for('login'))
    user_reports = DetectionReport.query.filter_by(user_id=session['user_id']).all()
    
    label_counts = {'AI_Generated': 0, 'from_GPT4': 0, 'Plagiarized': 0}
    enhanced = 0
    for r in user_reports:
        res = json.loads(r.label_results)
        for k in label_counts:
            if res.get(k, 0) > 0.5: label_counts[k] += 1
        if r.is_processed: enhanced += 1

    fig = px.bar(x=list(label_counts.keys()), y=list(label_counts.values()), 
                 title="Multi-Label Detection Trends", template="plotly_dark")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('reports.html', graphJSON=graphJSON, total=len(user_reports), enhanced=enhanced)

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    app.run(debug=True)
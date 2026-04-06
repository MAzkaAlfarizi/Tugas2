from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('model.pkl')

def get_prediction_table():
    # Membuat rentang tahun 2026 - 2030 sesuai logika di notebook
    future_years = np.array(range(2026, 2031))
    results = []
    
    for year in future_years:
        # Konversi ke format numerik (seconds since epoch)
        date_str = f"{year}-01-01"
        date_numeric = pd.to_datetime([date_str]).astype(int) // 10**9
        
        # Prediksi
        pred_value = model.predict(date_numeric.values.reshape(-1, 1))
        results.append({
            'tahun': year,
            'harga': round(pred_value[0], 2)
        })
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    tahun_input = None
    
    # Ambil data untuk tabel
    tabel_prediksi = get_prediction_table()
    
    # Metrik evaluasi model
    mae = 3.68
    mse = 25.70
    r2 = 0.55
    
    if request.method == 'POST':
        tahun_input = request.form.get('tahun')
        if tahun_input:
            date_str = f"{tahun_input}-01-01"
            date_numeric = pd.to_datetime([date_str]).astype(int) // 10**9
            pred_value = model.predict(date_numeric.values.reshape(-1, 1))
            prediction = round(pred_value[0], 2)

    return render_template('index.html', 
                           prediction=prediction, 
                           tahun=tahun_input, 
                           tabel_prediksi=tabel_prediksi,
                           mae=mae,
                           mse=mse,
                           r2=r2)

if __name__ == '__main__':
    app.run(debug=True)
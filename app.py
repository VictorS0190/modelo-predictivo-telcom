from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ============================
# CARGAR MODELO Y SCALER
# ============================

modelo = joblib.load('modelXGB.pkl')
scaler = joblib.load('scaler.pkl')


# ============================
# PAGINA PRINCIPAL
# ============================

@app.route('/')
def index():
    return render_template('index.html')


# ============================
# RESULTADOS
# ============================

@app.route('/resultados', methods=['POST'])
def resultados():

    try:

        # ============================
        # DATOS DEL FORM
        # ============================

        age = int(request.form['age'])
        tenure = int(request.form['tenure'])
        monthly = float(request.form['monthly'])
        contract = int(request.form['contract'])
        total = float(request.form['total'])
        internet = int(request.form['internet'])
        gender = int(request.form['gender'])
        support = int(request.form['support'])

        # ============================
        # DATAFRAME
        # ============================

        nuevo_cliente = pd.DataFrame([{
            "Age": age,
            "Tenure": tenure,
            "MonthlyCharges": monthly,
            "ContractType": contract,
            "TotalCharges": total,
            "InternetService_Fiber Optic": internet,
            "Gender_Male": gender,
            "TechSupport_Yes": support,
        }])

        # ============================
        # ESCALAR SOLO NUMERICAS
        # ============================

        columnas_numericas = [
            'Age',
            'Tenure',
            'MonthlyCharges',
            'TotalCharges'
        ]

        datos_numericos = scaler.transform(
            nuevo_cliente[columnas_numericas]
        )

        datos_numericos = pd.DataFrame(
            datos_numericos,
            columns=columnas_numericas
        )

        datos_numericos['ContractType'] = contract
        datos_numericos['InternetService_Fiber Optic'] = internet
        datos_numericos['Gender_Male'] = gender
        datos_numericos['TechSupport_Yes'] = support

        nuevo_cliente_final = datos_numericos[[
            'Age',
            'Tenure',
            'MonthlyCharges',
            'ContractType',
            'TotalCharges',
            'InternetService_Fiber Optic',
            'Gender_Male',
            'TechSupport_Yes'
        ]]

        # ============================
        # PREDICCION
        # ============================

        pred = modelo.predict(
            nuevo_cliente_final
        )[0]

        prob = modelo.predict_proba(
            nuevo_cliente_final
        )

        prob_renovar = round(
            prob[0][0] * 100,
            2
        )

        prob_cancelar = round(
            prob[0][1] * 100,
            2
        )

        # ============================
        # TEXTO RESULTADO
        # ============================

        if pred == 0:
            resultado = (
                "El cliente permanecerá "
                "en el servicio"
            )
        else:
            resultado = (
                "El cliente podría "
                "abandonar el servicio"
            )

        # ============================
        # NIVEL DE RIESGO
        # ============================

        if prob_cancelar < 40:

            riesgo = "BAJO"
            color = "#00ff4c"
            pdf = "bajo.pdf"

        elif prob_cancelar < 70:

            riesgo = "MEDIO"
            color = "#ffe600"
            pdf = "medio.pdf"

        else:

            riesgo = "ALTO"
            color = "#ff2b2b"
            pdf = "alto.pdf"

        return render_template(
            'resultados.html',

            resultado=resultado,

            prob_renovar=prob_renovar,
            prob_cancelar=prob_cancelar,

            riesgo=riesgo,
            color=color,

            pdf=pdf
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
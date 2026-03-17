from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib  #  Added to load the saved model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load and preprocess dataset (only for mapping, not for training)
df = pd.read_csv("C:\\Users\\binit\\OneDrive\\Desktop\\capstone\\yield_df.csv").dropna()
df['Crop_Type_Code'] = df['Item'].astype('category').cat.codes
df['Country_Code'] = df['Area'].astype('category').cat.codes
df['rain_pest_interaction'] = df['average_rain_fall_mm_per_year'] * df['pesticides_tonnes']
df['temp_rain_interaction'] = df['avg_temp'] * df['average_rain_fall_mm_per_year']


# Load pre-trained model 
model = joblib.load("models/crop_yield_rf_model.pkl")

# Generate crop and country mappings
crop_map = dict(zip(df['Item'].astype('category').cat.categories, range(len(df['Item'].astype('category').cat.categories))))
country_map = dict(zip(df['Area'].astype('category').cat.categories, range(len(df['Area'].astype('category').cat.categories))))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_generated = False
    selected_graph = 'graph.png'
    rainfall = ''
    pesticide = ''
    temperature = ''
    crop = list(crop_map.keys())[0]
    country = list(country_map.keys())[0]
    x_var = "Temperature"

    if request.method == 'POST':
        try:
            rainfall = float(request.form['rainfall'])
            pesticide = float(request.form['pesticide'])
            temperature = float(request.form['temperature'])
            crop = request.form['crop']
            country = request.form['country']
            x_var = request.form['x_var']

            rain_pest = rainfall * pesticide
            temp_rain = temperature * rainfall
            crop_code = crop_map[crop]
            country_code = country_map[country]

            input_df = pd.DataFrame([{
                'average_rain_fall_mm_per_year': rainfall,
                'pesticides_tonnes': pesticide,
                'avg_temp': temperature,
                'rain_pest_interaction': rain_pest,
                'temp_rain_interaction': temp_rain,
                'Crop_Type_Code': crop_code,
                'Country_Code': country_code
            }])
            prediction = model.predict(input_df)[0]

            # Generate x values for graph
            if x_var == 'Temperature':
                x_vals = np.linspace(5, 35, 100)
            elif x_var == 'Rainfall':
                x_vals = np.linspace(500, 2000, 100)
            elif x_var == 'Pesticide':
                x_vals = np.linspace(0, 150, 100)
            else:
                x_vals = np.linspace(0, 100, 100)

            # Predict yield across varying x
            y_preds = []
            for x in x_vals:
                temp = temperature if x_var != 'Temperature' else x
                rain = rainfall if x_var != 'Rainfall' else x
                pest = pesticide if x_var != 'Pesticide' else x

                X_plot = pd.DataFrame([{
                    'average_rain_fall_mm_per_year': rain,
                    'pesticides_tonnes': pest,
                    'avg_temp': temp,
                    'rain_pest_interaction': rain * pest,
                    'temp_rain_interaction': temp * rain,
                    'Crop_Type_Code': crop_code,
                    'Country_Code': country_code
                }])
                y_preds.append(model.predict(X_plot)[0])

            # Save plot
            plt.figure(figsize=(7, 4))
            plt.plot(x_vals, y_preds, label=f'Yield vs. {x_var}')
            plt.xlabel(x_var)
            plt.ylabel('Yield (hg/ha)')
            plt.title('Yield Prediction Curve')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig('static/graph.png')
            plt.close()
            image_generated = True

        except Exception as e:
            print("Prediction error:", e)

    return render_template('index.html',
                           crops=list(crop_map.keys()),
                           countries=list(country_map.keys()),
                           prediction=prediction,
                           image_generated=image_generated,
                           selected_graph=selected_graph,
                           graph_choice=x_var,
                           rainfall=rainfall,
                           pesticide=pesticide,
                           temperature=temperature,
                           selected_crop=crop,
                           selected_country=country)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script

MODEL_PATH = "gemstone_price_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
categorical_features = joblib.load(ENCODER_PATH)
expected_features = model.feature_names_in_.tolist()

print("Categorical Features:", categorical_features)
print("Model Expected Features:", expected_features)

if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame(
        {'Feature': model.feature_names_in_, 'Importance': model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    # Print each feature and its importance vertically
    for index, row in feature_importance.iterrows():
        print(f"{row['Feature']}: {row['Importance']}")
else:
    print("Feature importance not available for this model type.")

port = int(os.environ.get("PORT", 10000))
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])  # Convert JSON to DataFrame

        print("Original Input Data:\n", df)
        missing_cols = [col for col in categorical_features if col not in df.columns]
        print("Missing Categorical Columns:", missing_cols)
       
       
        # Apply One-Hot Encoding to categorical features
        df = pd.get_dummies(df, columns=categorical_features, drop_first=False)
        print(df)
        df = df.astype(int)

        df = df.reindex(columns=expected_features, fill_value=0)
        print("Not heated value of", df["Treatment_Not Heated"])
        # print("heated value", df["Treatment_Heated"])

        # Scaling the data
        numerical_features = ["Hardness", "Carat"]
        df[numerical_features] = scaler.transform(df[numerical_features])

        print("Transformed input data:\n", df)

        # Predict price
        prediction = model.predict(df)[0]
        return jsonify({'predicted_price': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
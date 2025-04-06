from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os


app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "gemstone_price_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")

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

port = int(os.environ.get("PORT", 5002))
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])  # Convert JSON to DataFrame
       
        # Apply One-Hot Encoding to categorical features
        df = pd.get_dummies(df, columns=categorical_features, drop_first=False)
        print(df)
        df = df.astype(int)# converting to integer in our case 0 ro 1
        df = df.reindex(columns=expected_features, fill_value=0)# aligning and checking the df features with expected feature, missing other features are 0
        print("Reindexed Data:", df.columns)

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
    app.run(host="0.0.0.0", port=5002)
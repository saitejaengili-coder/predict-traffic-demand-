import joblib
import numpy as np

model = joblib.load("traffic_model.pkl")

sample = np.array([[17, 31, 70, 2, 60, 90]])

prediction = model.predict(sample)

print("Predicted Vehicle Count:", int(prediction[0]))
import uvicorn
from fastapi import FastAPI 
import joblib
from os.path import dirname, join, realpath
from typing import List

app = FastAPI(
    title="Iris Prediction Model API",
    description="A simple API that use LogisticRegression model to predict the Iris species",
    version="0.1",
)


# load  model

with open(
    join(dirname(realpath(__file__)), "models/IrisClassifier.pkl"), "rb"
) as f:
    model = joblib.load(f)

def data_clean(str):
    arr = str.split(',')
    arr = list(map(float,arr))
    return arr

    #self = self.reshape(-1,1)
    

# Create Prediction Endpoint
@app.get("/predict-result")
def predict_iris(request):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    # perform prediction
    request = data_clean(request)
    prediction = model.predict([request])
    output = int(prediction[0])
    probas = model.predict_proba([request])
    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    species = {0: "Setosa", 1: "Versicolour", 2:"Virginica"}
    
    # show results
    result = {"prediction": species[output], "Probability": output_probability}
    return result

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8001)
#sepal_length=7.233
# sepal_width=4.652&petal_length=7.39&petal_width=0.324
#request.sepal_length,request.sepal_width,request.petal_length,request.petal_width
#7.233&4.652&7.39&0.324
from fastapi import FastAPI , UploadFile , File
from prediction import predict




app = FastAPI()

@app.get('/')
def read_root():
    return {"Message": "GO to /docs for the API testing"}


@app.post('/predict')
async def get_prediction(file: UploadFile = File(...)):  #  Pass file to predict()
    return await predict(file)  # CALL the function and await






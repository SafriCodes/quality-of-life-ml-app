QUALITY OF LIFE ML WEB APP

Files:
- Updated Quality of Life Data.csv: your dataset
- train_model.py: trains and saves both models; prints R² and accuracy
- app.py: FastAPI backend and two POST endpoints
- index.html: webpage with two forms
- requirements.txt: libraries needed

HOW TO RUN (in VS Code terminal)

1. Keep all these files in the same folder.
2. Install libraries:
   pip install -r requirements.txt

3. Train the models:
   python train_model.py

   It prints the R² and accuracy, then saves models in the models folder.

4. Start the website:
   uvicorn app:app --reload

5. Open this address in your browser:
   http://127.0.0.1:8000

ENDPOINTS
- POST /predict/regression
- POST /predict/classification

WHAT THE MODELS DO
- Regression predicts age_at_death using gender, occupation and daily hours.
- Classification predicts occupation_type using gender and daily hours.

NOTE
These are demonstration models. Predictions show patterns learned from this dataset and should not be treated as reliable real-life estimates of lifespan or occupation.

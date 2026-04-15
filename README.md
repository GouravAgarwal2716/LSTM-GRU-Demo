# LSTM/GRU Next-Word Prediction

This project implements a next-word prediction model using TensorFlow/Keras on the `hamlet.txt` dataset. The notebook trains a GRU-based language model, saves the model and tokenizer, and the Streamlit app loads those artifacts to make live predictions.

## Files

- `experiemnts.ipynb` — Jupyter notebook for preprocessing, training, and evaluation
- `app.py` — Streamlit application for interactive next-word prediction
- `hamlet.txt` — source text used for training
- `next_word_lstm.h5` — trained Keras model file
- `tokenizer.pickle` — saved tokenizer used for inference
- `requirements.txt` — Python package dependencies

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the environment:

- Windows PowerShell:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- Windows CMD:
  ```cmd
  .\venv\Scripts\activate.bat
  ```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Notebook

1. Launch Jupyter Notebook or JupyterLab from the project directory:

```bash
jupyter notebook
```

2. Open `experiemnts.ipynb`.
3. Run the notebook cells from top to bottom.
4. Save the model and tokenizer when prompted.

## Running the Streamlit App

Make sure `next_word_lstm.h5` and `tokenizer.pickle` are present in the project folder.

```bash
streamlit run app.py
```

Then open the local URL displayed in the terminal.

## Notes

- The app uses the saved model and tokenizer to predict the next word from a text input.
- If the notebook is retrained, rerun the save cell to update `next_word_lstm.h5` and `tokenizer.pickle`.
- The project currently uses TensorFlow 2.15 and Streamlit for the UI.

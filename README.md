# Tweet Sentiment Analysis with RNN

A Streamlit app that classifies the sentiment of a tweet as Negative, Neutral,
or Positive, using a recurrent neural network trained on tweet data.

## How it works

The model is an embedding layer (64-dimensional word vectors) followed by a
SimpleRNN layer (32 units) and a 3-way Dense output layer. Input tweets are
tokenized, padded to a fixed length of 166 tokens, and passed through the
trained model (`model.h5`) with its saved tokenizer (`tokenizer.pkl`).

The notebook `project.ipynb` walks through the ideas behind the model — why
text needs an embedding layer, how embeddings and RNNs work together, and how
the model is trained on `twitter_training.csv`.

## Features

- Text box for entering any tweet
- One-click sentiment prediction: Negative, Neutral, or Positive
- Pre-trained model and tokenizer included — no training needed to run the app

## Tech stack

- Python
- TensorFlow / Keras — RNN model
- Streamlit — web UI
- NumPy, pandas, scikit-learn — data handling

## Project structure

```
tweet-sentiment-rnn/
├── main.py               # Streamlit app
├── project.ipynb         # Model training and explanation notebook
├── model.h5              # Trained RNN model
├── tokenizer.pkl         # Fitted Keras tokenizer
├── twitter_training.csv  # Training data
└── requirements.txt      # Dependencies
```

## How to run

```bash
pip install -r requirements.txt
streamlit run main.py
```

Then open the local URL shown in the terminal, type a tweet, and click
**Predict Sentiment**.

## Notes

- A learning project built to understand embeddings and RNNs for text
  classification.

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to calculate perplexity
def calculate_perplexity(sentence, model, tokenizer):
    encoded_sentence = tokenizer.texts_to_sequences([sentence])
    padded_sentence = pad_sequences(encoded_sentence, maxlen=max_length, padding='post')

    # Get probabilities from the model (softmax output)
    predictions = model.predict(padded_sentence)[0]

    # Avoid log(0) by adding small value
    log_prob = np.log(predictions + 1e-10)

    # Calculate perplexity
    perplexity = np.exp(-np.sum(log_prob) / len(predictions))
    return perplexity

# Final prediction function
def classify_sentence(sentence):
    # Tokenize and pad the sentence
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Calculate perplexity
    perplexity = calculate_perplexity(sentence, model, tokenizer)

    # Scale perplexity
    perplexity_scaled = scaler.transform(np.array(perplexity).reshape(-1, 1))

    # Predict
    prediction = model.predict([padded_sequence, perplexity_scaled])[0][0]

    # Interpret result
    label = "AI-written" if prediction > 0.5 else "Human-written"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(f"Prediction: {label} (Confidence: {confidence:.2%}, Perplexity: {perplexity:.2f})")

classify_sentence("This sentence was definitely written by a human.")
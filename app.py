from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification , AutoModelForSeq2SeqLM


app=Flask(__name__)
# Load tokenizer and model for detection
tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-finetuned-fake-news-detection")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/bert-tiny-finetuned-fake-news-detection")

# Load tokenizer and model for summarization
summarization_tokenizer = AutoTokenizer.from_pretrained("it5/it5-large-news-summarization")
summarization_model = AutoModelForSeq2SeqLM.from_pretrained("it5/it5-large-news-summarization")

# Function to predict fake news
def detection_news(news_text):
    # Tokenize input
    inputs = tokenizer.encode_plus(news_text, return_tensors="pt", max_length=512, truncation=True)

    # Check for excessive input length
    if len(inputs["input_ids"][0]) > 512:
        raise ValueError("Input text is too long for the model to process.")

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits)

    return predicted_class.item(), torch.softmax(outputs.logits, dim=1)[0]


# Function to summarize news
def summarize_news(news_text):
    inputs = summarization_tokenizer.encode("summarize: " + news_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = summarization_model.generate(inputs, max_length=150, num_beams=2, length_penalty=2.0, early_stopping=True)
    return summarization_tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/predict', methods=['POST'])
def predict():
    data=request.get_json()
    news_text=data['news_text']
    summary = summarize_news(news_text)
    try:
        predicted_class, confidence = detection_news(news_text)
        if predicted_class == 1:
            prediction="label_0"
        else:
            prediction="label_1"
        return jsonify({'prediction' : prediction, 'confidence': confidence.tolist(), 'summary': summary})
    except ValueError as e:
        return jsonify({'error' : str(e)})


if __name__ == '__main__':
    app.run()

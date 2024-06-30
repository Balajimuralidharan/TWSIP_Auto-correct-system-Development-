import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from flask import Flask, request, jsonify
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.split()

# Example usage
text = "This is an example text with some errrors."
cleaned_text = preprocess(text)
print(cleaned_text)

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the dataset
def encode_data(texts, labels):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), torch.tensor(labels)

# Example data
texts = ["This is correct.", "Ths is incorret."]
labels = [1, 0]  # 1 for correct, 0 for incorrect
input_ids, attention_masks, labels = encode_data(texts, labels)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=(input_ids, attention_masks, labels)
)

# Train the model
trainer.train()



app = Flask(__name__)

@app.route('/correct', methods=['POST'])
def correct():
    text = request.json['text']
    preprocessed_text = preprocess(text)
    inputs = tokenizer(" ".join(preprocessed_text), return_tensors='pt')
    outputs = model(**inputs)
    _, predicted = torch.max(outputs.logits, dim=1)
    # Assuming 1 is correct and 0 is incorrect
    corrected_text = " ".join(preprocessed_text) if predicted.item() == 1 else "Corrected text here"
    return jsonify({'corrected_text': corrected_text})

if __name__ == '__main__':
    app.run(debug=True)

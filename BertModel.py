
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import logging
import transformers

class JobPostingClassifier():
    def __init__(self, model_path='model.pth'):
        
        # Suppress warnings
        transformers.logging.set_verbosity(logging.ERROR)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=4)        
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))  # Load weights on CPU
        self.model.eval()  # Set model to evaluation mode
        self.device = torch.device('cpu')  # Explicitly set device to CPU


    def predict(self, sentence):       
        
        # Prepare the sentence in the way the model expects
        encoding = self.tokenizer.encode_plus(
            sentence, 
            truncation=True, 
            padding='max_length', 
            max_length=512,
            return_tensors='pt'  # Return PyTorch tensors
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():  # Disabling gradient calculation
            outputs = self.model(input_ids, attention_mask=attention_mask)

        # The outputs are logits, we need to convert these to probabilities
        # by applying the softmax function
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # The result is a tensor where the first index is the batch size (1 in this case),
        # the second index is the label index, and the value is the probability.
        # We want the label with the highest probability, so we'll use argmax to find it
        predicted_label = probabilities.argmax().item()

        return predicted_label
    

# Pythia Quantized Model for Question/Answer

This repository hosts a quantized version of the Pythia model, fine-tuned for question/answer tasks. The model has been optimized for efficient deployment while maintaining high accuracy, making it suitable for resource-constrained environments.

## Model Details

- **Model Architecture:** Pythia-410m  
- **Task:** Chatbot  
- **Dataset:** sewon/ambig_qa  
- **Quantization:** Float16  
- **Fine-tuning Framework:** Hugging Face Transformers  

## Usage

### Installation

```sh
pip install transformers torch
```

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
model = AutoModelForCausalLM.from_pretrained("AventIQ-AI/pythia-410m-chatbot")

tokenizer.pad_token = tokenizer.eos_token

def chat_with_model(model, tokenizer, question, max_length=256):
    """Generate response to a question"""
    input_text = question
    
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  
            max_length=max_length,
            num_return_sequences=1,
            temperature=1.0,
            do_sample=True,  
            pad_token_id=tokenizer.pad_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
test_question = "What is the capital of France?"
response = chat_with_model(model, tokenizer, test_question)
print("Answer", response)
```

## Performance Metrics

- **Accuracy:** 0.56  
- **F1 Score:** 0.56  
- **Precision:** 0.68  
- **Recall:** 0.56  

## Fine-Tuning Details

### Dataset

The Hugging Face's `ambig_qa` dataset was used, containing both question and answer examples.

### Training

- Number of epochs: 3  
- Batch size: 4  
- Evaluation strategy: epoch  
- Learning rate: 2e-5  

### Quantization

Post-training quantization was applied using PyTorch's built-in quantization framework to reduce the model size and improve inference efficiency.

## Repository Structure

```
.
├── model/               # Contains the quantized model files
├── tokenizer/           # Tokenizer configuration and vocabulary files
├── model.safensors/     # Fine Tuned Model
├── README.md            # Model documentation
```

## Limitations

- The model may not generalize well to domains outside the fine-tuning dataset.  
- Quantization may result in minor accuracy degradation compared to full-precision models.  

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.


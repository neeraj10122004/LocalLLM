from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "distilgpt2"  # Small and efficient model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Generate text
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")  # Convert text to tensor
    with torch.no_grad():  # Disable gradient computation (saves memory)
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print("Generated Text:", generated_text)

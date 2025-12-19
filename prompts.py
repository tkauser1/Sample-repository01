# Best-performing prompt templates

# Text summarization prompt
summarize_prompt = "Summarize the following article in 100 words:"

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can also try "EleutherAI/gpt-neo-125M" for GPT-Neo
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Encode the input text (prompt)
prompt = "Summarize the following article in 100 words: [Insert Article Here]"
inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(inputs, max_length=150, num_return_sequences=1)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

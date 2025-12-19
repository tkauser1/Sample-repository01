from transformers import pipeline

# Initialize the pipeline with truncation
generator = pipeline("text-generation", model="gpt2", truncation=True)

# Example prompt for text generation
story_premise = "A young adventurer sets out on a journey to discover hidden treasure in an ancient city."

# Generate a story based on the prompt
generated_story = generator(story_premise, max_length=150, num_return_sequences=1)

# Print the generated story
print(f"Generated Story: {generated_story[0]['generated_text']}")
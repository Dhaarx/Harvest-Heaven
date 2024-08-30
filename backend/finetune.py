from transformers import T5ForConditionalGeneration, T5Tokenizer # type: ignore
import torch # type: ignore

# # Define your repository ID
# repo_id = "Dharine/my_finetuned_model"  # Replace with your Hugging Face username and model name

# # Load the model and tokenizer
# model = T5ForConditionalGeneration.from_pretrained(repo_id)
# tokenizer = T5Tokenizer.from_pretrained(repo_id)



# # Ensure the model is set to run on CPU
# model.to("cpu")

# save_directory = "cropbot"

# # Save the model
# model.save_pretrained(save_directory)

# # Save the tokenizer
# tokenizer.save_pretrained(save_directory)

load_directory="cropbot"

model = T5ForConditionalGeneration.from_pretrained(load_directory)
tokenizer = T5Tokenizer.from_pretrained(load_directory)


# Define your input question
input_question = "In growing Irish potatoes, row planting and use of holes, which one conserves fertilizers?"

# Preprocess the input question
inputs = tokenizer(input_question, return_tensors="pt", max_length=128, truncation=True)

# Generate the answer
with torch.no_grad():  # Disable gradient calculations for inference
    outputs = model.generate(inputs["input_ids"], max_new_tokens=50)

# Decode the generated tokens to get the answer
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Answer:", answer)

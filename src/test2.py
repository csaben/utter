import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# mp3_path =  "../input/2022_11_13_Carbon_14_Tears.mp3"
# model_path = "../models/base.en.pt"
# config_path = "../models/config.json"


#FIND A VIDEO THAT USES THE HUGGING FACE 6GB model locally. i dont want to use the api!!
model = GPT2LMHeadModel.from_pretrained()
tokenizer = GPT2Tokenizer.from_pretrained()

# Encode the prompt and generate text
prompt = "What is the capital of France?"
input_ids = torch.tensor(tokenizer.encode(prompt, return_tensors='pt')).unsqueeze(0)
output = model.generate(input_ids, max_length=1024, temperature=0.5)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)


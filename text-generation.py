from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.3,  # Increases randomness
        top_k=100,  # Avoids extremely unlikely words
        top_p=0.75,  # keeps only most probable words
        repetition_penalty=1.2,  # Reduces repetition
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


input_prompt = "One day I decided to learn LLM and"
generated_text = generate_text(input_prompt)
print("Generated Text:\n", generated_text)

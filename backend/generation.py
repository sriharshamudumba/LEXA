from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the TinyLlama model
print("Loading TinyLlama model... (this will take a few seconds)")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Move model to device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on device: {device}")

def generate_response(task: str, question: str, context: str) -> str:
    """
    Generate a smart answer using TinyLlama, without internet or API calls.
    """
    prompt = f"Task: {task}\nContext: {context}\nQuestion: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=400,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the answer part
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[1].strip()
    else:
        answer = generated_text.strip()

    return answer

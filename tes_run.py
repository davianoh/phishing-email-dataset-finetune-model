import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Load model ---
MODEL_PATH = "./exports/phishing-qwen3.5-2b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# --- System prompt (same as training) ---
SYSTEM_PROMPT = (
    "You are an email security analyst. Analyze the provided email and determine "
    "if it is a phishing attempt. Respond ONLY with a valid JSON object using this "
    'exact schema: {"is_phishing": boolean, "confidence_score": number (0.0-1.0), '
    '"threat_type": "string or null", "risk_level": "LOW|MEDIUM|HIGH|CRITICAL", '
    '"reasoning": "string"}'
)

# --- Inference function ---
def analyze_email(email_text: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": email_text}
    ]

    # Use the model's built-in chat template (important for Qwen3.5)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # matches your training config
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,  # extra room for thinking tokens
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only newly generated tokens
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    # Strip thinking block <think>...</think> if present, keep only final JSON

    try:
        result = json.loads(generated.strip())
    except json.JSONDecodeError:
        result = {"error": "Failed to parse model output", "raw": generated}

    return result


# --- Test it ---
if __name__ == "__main__":
    email = """Sender: security@netf1ix-account-alert.com
Receiver: kevin.walsh@hotmail.com
Subject: Netflix: Payment declined - Update your billing information
Body: Dear Kevin, We were unable to process your payment. 
Update billing: http://netf1ix-account-alert.com/billing-update. Act within 48 hours."""

    result = analyze_email(email)
    print(json.dumps(result, indent=2))
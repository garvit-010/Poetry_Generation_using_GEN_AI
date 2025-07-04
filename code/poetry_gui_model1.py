import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer


model_path = r"D:\Projects\Poetry_Generation_using_GEN_AI\code\poetry-gpt2-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def generate_poem(prompt, max_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=temperature
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

with gr.Blocks() as demo:
    gr.Markdown("# 📝 AI Poetry Generator")
    prompt = gr.Textbox(label="Enter your poetry prompt", lines=2, placeholder="e.g. Write a poem about the rain.")
    max_tokens = gr.Slider(20, 200, value=80, step=10, label="Max Tokens")
    temperature = gr.Slider(0.5, 1.5, value=0.9, step=0.05, label="Creativity (Temperature)")
    output = gr.Textbox(label="Generated Poem", lines=8)

    generate_btn = gr.Button("Generate Poem")
    generate_btn.click(
        generate_poem,
        inputs=[prompt, max_tokens, temperature],
        outputs=output
    )

demo.launch()

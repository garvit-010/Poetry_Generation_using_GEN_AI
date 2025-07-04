import gradio as gr
import ollama

MODEL_NAME = r"hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

def generate_poem(prompt, max_tokens, temperature):
    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={
            "num_predict": int(max_tokens),
            "temperature": float(temperature)
        }
    )
    return response['response']

with gr.Blocks() as demo:
    gr.Markdown("# 📝 Ollama Poetry Generator")
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

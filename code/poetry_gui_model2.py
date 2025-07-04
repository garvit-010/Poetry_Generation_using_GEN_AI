import gradio as gr
import ollama

MODEL_NAME = r"hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"  # Use the local Ollama model name

def generate_poem(user_prompt, max_tokens, temperature):
    # Add the system prompt to instruct the model to always respond as a Hindi poet
    system_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a poet who always writes poems in Hindi.\n<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
    )
    assistant_tag = "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    full_prompt = system_prompt + user_prompt + assistant_tag

    response = ollama.generate(
        model=MODEL_NAME,
        prompt=full_prompt,
        options={
            "num_predict": int(max_tokens),
            "temperature": float(temperature)
        }
    )
    return response['response']

with gr.Blocks() as demo:
    gr.Markdown("# 📝 Ollama Hindi Poetry Generator")
    prompt = gr.Textbox(
        label="Enter your poetry prompt (in English)",
        lines=2,
        placeholder="e.g. Write a poem about the rain."
    )
    max_tokens = gr.Slider(20, 200, value=100, step=10, label="Max Tokens")
    temperature = gr.Slider(0.5, 1.5, value=0.9, step=0.05, label="Creativity (Temperature)")
    output = gr.Textbox(label="Generated Hindi Poem", lines=8)

    generate_btn = gr.Button("Generate Poem")
    generate_btn.click(
        generate_poem,
        inputs=[prompt, max_tokens, temperature],
        outputs=output
    )

demo.launch()

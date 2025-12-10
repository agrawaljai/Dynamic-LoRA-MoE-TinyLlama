# app.py

import gradio as gr
from pipeline import MoE_Pipeline

try:
    print("Initializing the MoE Pipeline for the demo...")
    pipeline = MoE_Pipeline()
    print("Pipeline ready.")
except FileNotFoundError as e:
    print(str(e))
    print("Error: Adapters not found. Please run train.py to train the models before launching the demo.")
    pipeline = None

def get_response(prompt):
    if pipeline is None:
        return "ERROR: Models not trained. Please run `python train.py` in your terminal."
    response, expert_name = pipeline.generate(prompt)
    output = f"**🤖 Expert Used:** `{expert_name}`\n\n---\n\n{response}"
    return output

iface = gr.Interface(
    fn=get_response,
    inputs=gr.Textbox(lines=5, label="Your Prompt", placeholder="e.g., Write a python function to... OR Tell me a story about..."),
    outputs=gr.Markdown(label="Model Response"),
    title="Dynamic Task Routing with LoRA Mixture of Experts",
    description="Enter a prompt and the system will route it to the correct expert (Code or Poetry). (Based on TinyLlama-1.1B)",
    examples=[
        ["Write a Python class for a singly linked list"],
        ["Compose a haiku about a rainy morning"]
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    print("Launching Gradio demo... Open the URL in your browser.")
    iface.launch()
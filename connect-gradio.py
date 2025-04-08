-- Claude

import gradio as gr
from typing import List
import os
from PIL import Image
import base64
from io import BytesIO

# Sample logo function (you can replace with your actual logo)
def create_sample_logo(text, color="#008080", size=(200, 70), bg_color="transparent"):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGBA', size, color=0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
    
    text_width, text_height = draw.textsize(text, font=font)
    position = ((size[0] - text_width) / 2, (size[1] - text_height) / 2)
    draw.text(position, text, font=font, fill=color)
    
    # Convert to base64 string
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return buffered.getvalue()

# Create the RAG pipeline function (placeholder)
def rag_pipeline(message: str):
    # This is where your actual RAG implementation would go
    return f"HR-Connect response: {message}"

# Chat function
def chat(message: str, history: List[List[str]]):
    response = rag_pipeline(message)
    return response

# Custom CSS for teal theme
custom_css = """
:root {
    --teal-primary: #008080;
    --teal-secondary: #006666;
    --teal-light: #99cccc;
    --teal-very-light: #e6f2f2;
}

.dark {
    --teal-primary: #00a0a0;
    --teal-secondary: #00b3b3;
    --teal-light: #004d4d;
    --teal-very-light: #002e2e;
}

/* Header styling */
#header-container {
    background-color: var(--teal-very-light);
    border-bottom: 2px solid var(--teal-primary);
    padding: 10px 20px;
    display: flex;
    align-items: center;
    margin-bottom: 20px;
    border-radius: 8px;
}

#header-image {
    margin-right: 15px;
}

#header-text {
    color: var(--teal-primary);
    font-size: 1.8em;
    font-weight: bold;
}

/* Footer styling */
#footer-container {
    background-color: var(--teal-very-light);
    border-top: 2px solid var(--teal-primary);
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 20px;
    border-radius: 8px;
}

#footer-text {
    color: var(--teal-primary);
    font-size: 0.9em;
}

/* Chat styling */
.chatbot-container .message.user {
    background-color: var(--teal-light);
}

.chatbot-container .message.bot {
    background-color: var(--teal-very-light);
    border-left: 4px solid var(--teal-primary);
}

#chatbot-textbox {
    border: 2px solid var(--teal-primary);
    border-radius: 8px;
}

#chatbot-textbox:focus {
    border-color: var(--teal-secondary);
    box-shadow: 0 0 0 2px var(--teal-light);
}

/* Button styling */
.primary-button {
    background-color: var(--teal-primary) !important;
}

.primary-button:hover {
    background-color: var(--teal-secondary) !important;
}

/* Theme toggle */
#theme-toggle {
    position: absolute;
    top: 20px;
    right: 20px;
    z-index: 1000;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    # Theme toggle
    with gr.Row(elem_id="theme-toggle"):
        theme_toggle = gr.Button("🌙 Dark Mode", scale=0)
    
    # Header
    with gr.Row(elem_id="header-container"):
        with gr.Column(scale=1, min_width=100):
            gr.HTML('<div id="header-image">🏢</div>')
        with gr.Column(scale=5):
            gr.HTML('<div id="header-text">🤝 HR-Connect</div>')
    
    # Chat interface
    chatbot = gr.ChatInterface(
        fn=chat,
        title=None,
        description="Ask HR-related questions and get answers with enhanced retrieval capabilities",
        examples=[
            "What is our company's parental leave policy?",
            "How do I request time off in the system?",
            "Tell me about our health insurance options"
        ],
        additional_inputs=None,
        submit_btn="Send 📤",
        retry_btn="Retry 🔄",
        undo_btn="Undo ↩️",
        clear_btn="Clear 🧹"
    )
    
    # Footer
    with gr.Row(elem_id="footer-container"):
        with gr.Column(scale=5):
            gr.HTML('<div id="footer-text">Powered by floridablue.com</div>')
        with gr.Column(scale=1, min_width=100):
            gr.HTML('<div id="footer-image">🔐</div>')
    
    # JavaScript for theme toggle functionality
    theme_js = """
    function toggleTheme() {
        document.body.classList.toggle('dark');
        const button = document.querySelector('#theme-toggle button');
        if (document.body.classList.contains('dark')) {
            button.textContent = '☀️ Light Mode';
        } else {
            button.textContent = '🌙 Dark Mode';
        }
    }
    
    document.querySelector('#theme-toggle button').addEventListener('click', toggleTheme);
    """
    
    demo.load(None, js=theme_js)

if __name__ == "__main__":
    demo.launch()

====  deepseek

import gradio as gr
from gradio.themes import Soft
import base64

# Custom CSS for styling
custom_css = """
/* Teal theme colors */
:root {
    --teal-primary: #008080;
    --teal-secondary: #4da6a6;
    --teal-light: #e6f2f2;
}

/* Header styling */
.header {
    padding: 1rem;
    background: var(--teal-primary);
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 1rem;
    color: white !important;
}

/* Footer styling */
.footer {
    padding: 1rem;
    background: var(--teal-primary);
    border-radius: 8px;
    text-align: center;
    color: white !important;
    margin-top: 1rem;
}

/* Chat message styling */
.gradio-container {
    background: var(--teal-light) !important;
}

.dark .gradio-container {
    background: #1a1a1a !important;
}

/* Input box styling */
.textbox {
    border: 2px solid var(--teal-secondary) !important;
    border-radius: 8px !important;
}

/* Button styling */
button {
    background: var(--teal-primary) !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
}

button:hover {
    background: var(--teal-secondary) !important;
}

/* Chat bubble styling */
.user, .assistant {
    padding: 8px 16px;
    border-radius: 20px !important;
    margin: 4px 0 !important;
}
"""

# Base64 encoded placeholder images (replace with your actual images)
header_img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
footer_img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."

def create_header():
    return gr.HTML(f"""
    <div class="header">
        <img src="{header_img}" height="40" alt="HR Connect Logo">
        <h1 style="margin: 0;">HR-Connect</h1>
        <div style="flex-grow: 1;"></div>
        <i class="fa fa-moon" id="theme-toggle"></i>
    </div>
    """)

def create_footer():
    return gr.HTML(f"""
    <div class="footer">
        <img src="{footer_img}" height="20" alt="Florida Blue Logo">
        <span style="margin-left: 8px;">Powered by floridablue.com</span>
    </div>
    """)

def toggle_theme(theme):
    return Soft() if theme == "light" else gr.themes.Dark()

def chat(message: str, history):
    response = rag_pipeline(message)
    return response

with gr.Blocks(theme=Soft(), css=custom_css) as demo:
    create_header()
    
    gr.ChatInterface(
        fn=chat,
        title="",
        description="Ask HR-related questions and get instant answers",
        examples=[
            ["How do I update my benefits?", "user"],
            ["What's our PTO policy?", "user"],
            ["Explain our health insurance options", "user"]
        ],
        chatbot=gr.Chatbot(
            bubble_full_width=False,
            show_copy_button=True,
            avatar_images=("user.png", "assistant.png"),
            render=False
        ),
        textbox=gr.Textbox(placeholder="Type your HR question here...", 
                         show_label=False,
                         container=False),
        submit_btn=gr.Button("Ask HR", variant="primary"),
        retry_btn=None,
        undo_btn=None,
        clear_btn=gr.Button("Clear Chat"),
    )
    
    create_footer()
    
    # Theme toggle
    theme = gr.Radio(["light", "dark"], value="light", visible=False)
    theme_btn = gr.Button("🌙", elem_id="theme-toggle")
    theme_btn.click(
        fn=lambda t: "dark" if t == "light" else "light",
        inputs=theme,
        outputs=theme,
    ).then(
        None,
        inputs=theme,
        _js="(theme) => document.body.classList.toggle('dark', theme === 'dark')"
    )

# Add Font Awesome for icons
demo.head = f"""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
    #theme-toggle {{cursor: pointer; padding: 8px;}}
    .dark {{background: #1a1a1a; color: white;}}
</style>
"""

if __name__ == "__main__":
    demo.launch()

    ===== O1 preview 

    import gradio as gr
from PIL import Image

# Assuming you have these images in your directory
header_image_path = "header_image.png"
footer_image_path = "footer_image.png"

# Function to load images
def load_image(path):
    return Image.open(path)

# Chat function
def chat(message: str, history: List[List[str]]):
    response = rag_pipeline(message)
    return response

# Custom CSS for styling
custom_css = """
:root {
    --teal-color: #008080;
}

body {
    background-color: var(--teal-color);
}

#chatbot {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
}

#chatbot .message.user {
    background-color: var(--teal-color);
    color: white;
}

#chatbot .message.bot {
    background-color: #e0e0e0;
}

#chatbot input {
    background-color: var(--teal-color);
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px;
}

#chatbot .footer {
    background-color: var(--teal-color);
    color: white;
    text-align: center;
    padding: 10px;
}

#chatbot .header {
    background-color: var(--teal-color);
    color: white;
    text-align: center;
    padding: 10px;
}

.dark-mode {
    background-color: #333;
    color: white;
}

.dark-mode #chatbot {
    background-color: #444;
}

.dark-mode #chatbot .message.user {
    background-color: #006666;
}

.dark-mode #chatbot .message.bot {
    background-color: #555;
}

.dark-mode #chatbot input {
    background-color: #006666;
}

.dark-mode .footer {
    background-color: #006666;
}

.dark-mode .header {
    background-color: #006666;
}
"""

# Gradio Interface
with gr.Blocks(css=custom_css, theme="soft") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            header_image = gr.Image(value=load_image(header_image_path), label=None)
        with gr.Column(scale=3):
            gr.Markdown("# HR-Connect", elem_classes="header")

    chatbot = gr.Chatbot([], elem_id="chatbot")
    msg = gr.Textbox(label="Type your message here", elem_id="chat_input")
    clear = gr.Button("Clear")

    with gr.Row():
        with gr.Column(scale=1):
            footer_image = gr.Image(value=load_image(footer_image_path), label=None)
        with gr.Column(scale=3):
            gr.Markdown("Powered by floridablue.com", elem_classes="footer")

    with gr.Row():
        dark_mode_toggle = gr.Checkbox(label="Dark Mode", value=False)

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

    def toggle_dark_mode(is_dark):
        if is_dark:
            return {"_js": "() => {document.body.classList.add('dark-mode');}"}
        else:
            return {"_js": "() => {document.body.classList.remove('dark-mode');}"}

    dark_mode_toggle.change(toggle_dark_mode, dark_mode_toggle)

# Launch the interface
demo.launch()

==== 

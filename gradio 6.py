-- deepseek : 

import gradio as gr

# Suggested questions update logic
def update_suggestions(history):
    last_message = history[-1][0].lower() if history else ""
    suggestions = [
        "What is our PTO policy?",
        "How does health insurance work?",
        "What are the company holidays?",
        "How do I request vacation time?"
    ]
    
    if "insurance" in last_message:
        suggestions = [
            "What's covered in dental insurance?",
            "When does health insurance start?",
            "Can I add dependents?",
            "What's the vision care coverage?"
        ]
    elif "pto" in last_message or "vacation" in last_message:
        suggestions = [
            "How much PTO do I accrue?",
            "Can I carry over PTO?",
            "What's the approval process?",
            "How to check my balance?"
        ]
    
    return [gr.update(visible=True, value=s) for s in suggestions] + [gr.update(visible=False)]*(4-len(suggestions))

# Dummy response generator for demo
def respond(message, history, model):
    responses = {
        "llama": f"Llama: Company policy states... (Response to: {message})",
        "mistral": f"Mistral: According to our handbook... (Response to: {message})",
        "mixtral": f"Mixtral: HR guidelines indicate... (Response to: {message})"
    }
    return responses.get(model, "Please select a valid model")

# Update model display
def update_model_display(model):
    return f"Current Model: {model}"

# Custom CSS for additional styling
custom_css = """
footer {visibility: hidden}
.header {
    text-align: center;
    padding: 20px;
    background-color: #008080;
    color: white;
    border-radius: 5px;
}
.footer {
    text-align: center;
    padding: 10px;
    background-color: #008080;
    color: white;
    border-radius: 5px;
    margin-top: 20px;
}
"""

with gr.Blocks(theme=gr.themes.Default(
    primary_hue="teal", 
    secondary_hue="teal",
    neutral_hue="slate"
), css=custom_css) as demo:
    # Header
    gr.Markdown("""<div class="header">
                    <h1>🤖 HR Assistant</h1>
                    <p>Ask about company policies, benefits, and HR-related questions</p>
                   </div>""")
    
    # Model selection
    with gr.Row():
        model_selector = gr.Dropdown(
            choices=["llama", "mistral", "mixtral"],
            value="llama",
            label="Select AI Model"
        )
        model_display = gr.Text("Current Model: llama", interactive=False)
    
    # Chat interface
    chatbot = gr.ChatInterface(
        respond,
        chatbot=gr.Chatbot(height=400),
        additional_inputs=[model_selector]
    )
    
    # Suggested questions
    with gr.Row() as suggestion_row:
        suggestion_btns = [
            gr.Button("Suggested Question 1", visible=False),
            gr.Button("Suggested Question 2", visible=False),
            gr.Button("Suggested Question 3", visible=False),
            gr.Button("Suggested Question 4", visible=False)
        ]
    
    # Footer
    gr.Markdown("""<div class="footer">
                    <p>© 2024 Company Name. Confidential HR Assistant</p>
                   </div>""")
    
    # Update model display when selection changes
    model_selector.change(
        update_model_display,
        inputs=model_selector,
        outputs=model_display
    )
    
    # Update suggestions after chat interaction
    chatbot.chatbot.change(
        update_suggestions,
        inputs=chatbot.chatbot,
        outputs=suggestion_btns
    )
    
    # Connect suggestion buttons to chat input
    for btn in suggestion_btns:
        btn.click(
            lambda x: x,
            inputs=btn,
            outputs=chatbot.textbox
        )

if __name__ == "__main__":
    demo.launch()

======== o1 preview 

import gradio as gr

def chat_with_model(message, history, model):
    # Simulated response from the selected model (for demonstration purposes)
    response = f"Response from {model}: {message[::-1]}"  # Reverses the message as a placeholder
    history = history + [(message, response)]
    return history, f"Current Model: {model}"

def update_suggested_questions(message, suggested_questions):
    # Generates a new suggested question based on the user's message
    suggested_question = f"What else would you like to know about '{message}'?"
    suggested_questions = suggested_questions + [suggested_question]
    suggested_questions_display = '\n'.join(suggested_questions)
    return suggested_questions, suggested_questions_display

def update_model(selected_model):
    # Updates the displayed current model
    return f"Current Model: {selected_model}"

with gr.Blocks(theme=gr.themes.Default(primary_hue="teal")) as demo:
    # Header
    gr.Markdown("<h1 style='text-align: center;'>HR Chatbot</h1>")
    gr.Markdown("---")
    
    # Model selection and current model display
    with gr.Row():
        model = gr.Dropdown(
            choices=['llama', 'mistral', 'mixtral'],
            label='Select Model',
            value='llama'
        )
        current_model = gr.Textbox(
            value="Current Model: llama",
            label="",
            interactive=False
        )
    
    # Chat interface
    chatbot = gr.Chatbot(label="Chat with HR Bot")
    msg = gr.Textbox(label="Your Message")
    send = gr.Button("Send")
    
    # Suggested questions
    suggested_questions = gr.State([])
    suggested_questions_display = gr.Textbox(
        label="Suggested Questions",
        interactive=False,
        lines=5
    )

    # Function to handle user input
    def user_input(user_message, chat_history, model_selection, suggested_questions_list):
        chat_history, current_model_value = chat_with_model(user_message, chat_history, model_selection)
        suggested_questions_list, suggested_questions_text = update_suggested_questions(user_message, suggested_questions_list)
        return chat_history, current_model_value, suggested_questions_list, suggested_questions_text

    # Bind the send button to the user_input function
    send.click(
        user_input,
        inputs=[msg, chatbot, model, suggested_questions],
        outputs=[chatbot, current_model, suggested_questions, suggested_questions_display]
    )
    
    # Update current model display when model selection changes
    model.change(
        update_model,
        inputs=model,
        outputs=current_model
    )

    # Footer
    gr.Markdown("---")
    gr.Markdown("<p style='text-align: center;'>© 2025 HR Chatbot</p>")

demo.launch()

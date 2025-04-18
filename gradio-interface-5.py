import gradio as gr
import os
import requests
import json
import random
from typing import Dict, List, Optional, Union, Any

class MistralLLM:
    """
    A custom LLM class for using Mistral models via the Hugging Face Inference API.
    """
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        api_token: Optional[str] = None,
        api_url: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        """Initialize the MistralLLM."""
        self.model_name = model_name
        
        # Get API token from env variable if not provided
        self.api_token = api_token or os.environ.get("HF_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "API token must be provided either as an argument or as an environment variable 'HF_API_TOKEN'"
            )
        
        # Set up API URL
        self.api_url = api_url or f"https://api-inference.huggingface.co/models/{model_name}"
        
        # Generation parameters
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.additional_params = kwargs
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
    
    def _prepare_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Prepare the payload for the API request."""
        # Start with default parameters
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "do_sample": True,
                "return_full_text": False,  # Only return the generated text, not the prompt
            }
        }
        
        # Add any additional parameters from init or call
        for k, v in {**self.additional_params, **kwargs}.items():
            if k not in ["max_tokens", "temperature", "top_p"]:
                payload["parameters"][k] = v
                
        return payload
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text based on the prompt."""
        payload = self._prepare_payload(prompt, **kwargs)
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            data=json.dumps(payload)
        )
        
        if response.status_code != 200:
            error_msg = f"API request failed with status code {response.status_code}: {response.text}"
            raise Exception(error_msg)
            
        return response.json()
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text and return just the generated text string."""
        response = self.generate(prompt, **kwargs)
        
        # Handle different response formats from Hugging Face
        if isinstance(response, list) and len(response) > 0:
            if "generated_text" in response[0]:
                return response[0]["generated_text"]
            return response[0]
        elif isinstance(response, dict) and "generated_text" in response:
            return response["generated_text"]
        
        # Fallback
        return str(response)
    
    def update_temperature(self, temperature: float):
        """Update the temperature parameter."""
        self.temperature = temperature

    def __call__(self, prompt: str, **kwargs) -> str:
        """Make the class callable for easy use"""
        return self.generate_text(prompt, **kwargs)


class MockLLM:
    """A mock LLM class for demo purposes when no API token is available."""
    
    def __init__(self, model_name="mock-model", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        
        # Different response patterns based on model name
        self.model_responses = {
            "mistralai/Mistral-7B-Instruct-v0.2": {
                "style": "concise and informative",
                "signature": "- Mistral 7B"
            },
            "mistralai/Mixtral-8x7B-Instruct-v0.1": {
                "style": "detailed and comprehensive",
                "signature": "- Mixtral 8x7B"
            },
            "mistralai/Mistral-7B-v0.1": {
                "style": "straightforward and clear",
                "signature": "- Mistral 7B v0.1"
            },
            "mock-model": {
                "style": "basic",
                "signature": "- Mock Model"
            }
        }
        
        # Get the response style for the specified model or use default
        self.response_style = self.model_responses.get(
            model_name, 
            self.model_responses["mock-model"]
        )
    
    def generate_text(self, prompt, **kwargs):
        # Override instance temperature if provided in kwargs
        temperature = kwargs.get("temperature", self.temperature)
        
        # Generate suggestions if requested
        if "follow-up questions" in prompt:
            return self._generate_suggestions(temperature)
        
        # Generate response based on query content
        response = self._generate_response(prompt, temperature)
        return response
    
    def _generate_suggestions(self, temperature):
        """Generate mock suggested questions with temperature influence."""
        # Base suggestions
        if "Mixtral" in self.model_name:
            base_suggestions = [
                "1. What are the key differences between Mixtral and other models?",
                "2. How does Mixtral perform on reasoning tasks?",
                "3. What technical innovations make Mixtral unique?",
                "4. How can I fine-tune Mixtral for my specific application?",
                "5. What are the computational requirements for running Mixtral locally?"
            ]
            # Alternative suggestions for variety with higher temperature
            alt_suggestions = [
                "1. How does Mixtral's mixture of experts architecture work?",
                "2. Can you explain Mixtral's context window capabilities?",
                "3. What are some creative applications of Mixtral?",
                "4. How does Mixtral compare to other open source models?",
                "5. What are the ethical considerations when deploying Mixtral?"
            ]
        elif "Mistral" in self.model_name:
            base_suggestions = [
                "1. How does Mistral compare to other 7B models?",
                "2. What are the best use cases for Mistral?",
                "3. Can you explain how Mistral handles context?",
                "4. What are some limitations of Mistral?",
                "5. How can I optimize prompts for Mistral?"
            ]
            # Alternative suggestions for variety with higher temperature
            alt_suggestions = [
                "1. What makes Mistral's architecture efficient?",
                "2. How well does Mistral perform on coding tasks?",
                "3. What are the differences between Mistral versions?",
                "4. Can Mistral be run on consumer hardware?",
                "5. What creative tasks can Mistral excel at?"
            ]
        else:
            base_suggestions = [
                "1. How does deep learning differ from traditional machine learning?",
                "2. What are transformers in NLP?",
                "3. Can you explain how language models work?",
                "4. What are some ethical concerns with AI?",
                "5. How can I start learning AI as a beginner?"
            ]
            # Alternative suggestions for variety with higher temperature
            alt_suggestions = [
                "1. How might AI development change in the next decade?",
                "2. What are some unconventional applications of NLP?",
                "3. Can you explain the concept of AI alignment?",
                "4. How do different cultures perceive AI development?",
                "5. What creative fields are being transformed by AI?"
            ]
        
        # Use temperature to determine mix of base and alternative suggestions
        # Higher temperature = more alternative/creative suggestions
        if temperature <= 0.3:
            return "\n".join(base_suggestions)
        elif temperature <= 0.7:
            # Mix some base and some alternative
            mixed = []
            for i in range(5):
                if random.random() < temperature:
                    mixed.append(alt_suggestions[i])
                else:
                    mixed.append(base_suggestions[i])
            return "\n".join(mixed)
        else:
            # Higher temperature uses more alternative suggestions
            return "\n".join(alt_suggestions)
    
    def _generate_response(self, prompt, temperature):
        """Generate a mock response based on the prompt and temperature."""
        style = self.response_style["style"]
        signature = self.response_style["signature"]
        
        # Extract user query from the prompt
        query = prompt.split("User: ")[-1].split("\nAssistant:")[0]
        
        # Base responses for common queries
        if "machine learning" in query.lower():
            base_response = f"Machine learning is a subset of artificial intelligence that involves algorithms that can learn from and make predictions based on data. As a {style} model, I'd highlight that it uses statistical methods to enable computers to improve their performance on a task through experience."
            detailed_response = f"Machine learning is a branch of artificial intelligence focused on building systems that learn from data. Unlike traditional programming where explicit instructions are given, ML algorithms identify patterns in data and make decisions with minimal human intervention. There are several types including supervised learning (using labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error). These approaches power applications from recommendation systems to computer vision and natural language processing. The field continues to evolve with deep learning techniques enabling breakthroughs in previously challenging domains."
        
        elif "neural network" in query.lower():
            base_response = f"Neural networks are computing systems vaguely inspired by the biological neural networks in animal brains. They use interconnected nodes (neurons) organized in layers to process information. This {style} explanation focuses on their ability to learn complex patterns from data through adjusting connection weights."
            detailed_response = f"Neural networks are computational models inspired by the human brain's structure and function. They consist of interconnected nodes (neurons) organized in layers - typically an input layer, one or more hidden layers, and an output layer. Each connection between neurons carries a weight that adjusts during learning. Information flows forward (feedforward networks) or in loops (recurrent networks), with activation functions determining how signals propagate. Deep learning uses neural networks with many layers to learn hierarchical representations of data. This architecture has revolutionized computer vision, NLP, and other fields by enabling machines to learn complex patterns directly from raw data without manual feature engineering."
        
        elif "language model" in query.lower() or "llm" in query.lower():
            base_response = f"Language models are AI systems trained to understand and generate human language. They learn patterns from vast amounts of text data and can generate coherent responses to prompts. This {style} description notes that modern LLMs use transformer architectures to process contextual relationships between words."
            detailed_response = f"Language models are AI systems trained to understand, process, and generate human language. They work by analyzing patterns in vast text corpora to predict probable word sequences. Modern Large Language Models (LLMs) typically use transformer architectures with attention mechanisms to process text bidirectionally, capturing nuanced contextual relationships. These models learn grammar, facts, reasoning patterns, and even cultural context during pretraining, then may undergo further refinement through fine-tuning and alignment techniques. While powerful in generating coherent text and solving various language tasks, they face challenges including hallucinations, bias reproduction, and contextual limitations. Models vary significantly in size, architecture, training data, and specialization, with organizations developing both closed and open-source variants optimized for different applications."
        
        else:
            base_response = f"I'm providing this {style} response as a demonstration of the {self.model_name} interface. In a real implementation, this would be a genuine response from the selected model. For more specific information, please ask a more detailed question."
            detailed_response = f"I'm providing this {style} response as a demonstration of the {self.model_name} interface. This expanded explanation simulates how language models generate more diverse and elaborate responses at higher temperature settings. With increased temperature, outputs typically show greater linguistic variety, creative expression, and sometimes unexpected connections between concepts. In a real implementation with the actual model, you would see more nuanced differences based on the temperature parameter, with lower values producing more deterministic, focused responses and higher values encouraging exploration of different possibilities in the response space. For more specific information on any topic, please feel free to ask a more detailed question."
        
        # Apply temperature effect to response verbosity and style
        if temperature <= 0.3:
            # Low temperature - concise, focused response
            response = base_response
        elif temperature <= 0.7:
            # Medium temperature - moderate detail
            response = base_response + " " + detailed_response.split(". ")[0] + "."
        else:
            # High temperature - verbose, detailed response
            response = detailed_response
            
            # Add some creative flair for very high temperatures
            if temperature > 0.9:
                creative_additions = [
                    f" This perspective illustrates how {query.lower().split()[0] if len(query.split()) > 0 else 'concepts'} interconnect with broader technological and societal themes.",
                    f" Exploring this topic further reveals fascinating implications for future research directions.",
                    f" This represents just one viewpoint in an evolving conceptual landscape.",
                    f" The intersection of these ideas with other domains creates opportunities for innovative applications."
                ]
                response += random.choice(creative_additions)
        
        # Always add the model signature
        return response + " " + signature
    
    def update_temperature(self, temperature: float):
        """Update the temperature parameter."""
        self.temperature = temperature


class ChatbotWithSuggestions:
    """
    A chatbot class that provides suggested questions and maintains chat history.
    """
    def __init__(self, llm, initial_suggestions=None):
        """
        Initialize the chatbot with an LLM and optional initial suggestions.
        
        Args:
            llm: An instance of MistralLLM or similar text generation model
            initial_suggestions: List of initial suggested questions
        """
        self.llm = llm
        self.chat_history = []
        
        # Default initial suggestions if none provided
        self.initial_suggestions = initial_suggestions or [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain language models",
            "What are the applications of AI in healthcare?",
            "What is the difference between AI and ML?"
        ]
        
        self.current_suggestions = self.initial_suggestions.copy()
    
    def generate_response(self, user_input):
        """Generate a response to the user input."""
        # Create a prompt that includes chat history for context
        history_prompt = ""
        if self.chat_history:
            for role, text in self.chat_history:
                if role == "user":
                    history_prompt += f"User: {text}\n"
                else:
                    history_prompt += f"Assistant: {text}\n"
        
        prompt = f"{history_prompt}User: {user_input}\nAssistant:"
        
        # Generate response using the LLM
        try:
            response = self.llm.generate_text(prompt, max_tokens=512)
            self.chat_history.append(("user", user_input))
            self.chat_history.append(("assistant", response))
            return response
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            return error_message
    
    def generate_suggestions(self, user_input=None, last_response=None):
        """
        Generate suggested follow-up questions based on the conversation history.
        
        Args:
            user_input: The latest user input if available
            last_response: The latest bot response if available
        
        Returns:
            List of suggested questions
        """
        # If no conversation history yet, return initial suggestions
        if not self.chat_history:
            return self.initial_suggestions
        
        # Create a prompt to generate relevant follow-up questions
        if last_response is None and self.chat_history:
            # Get the last assistant response from history
            last_user_inputs = [text for role, text in self.chat_history if role == "user"]
            last_responses = [text for role, text in self.chat_history if role == "assistant"]
            
            if last_user_inputs and last_responses:
                user_input = last_user_inputs[-1]
                last_response = last_responses[-1]
        
        suggestion_prompt = f"""
Based on the following conversation, generate 5 relevant follow-up questions the user might ask.
Make them diverse but related to the topic. Format as a numbered list.

User: {user_input}
Assistant: {last_response}

Suggested follow-up questions:
"""
        
        try:
            suggestions_text = self.llm.generate_text(suggestion_prompt, max_tokens=200)
            
            # Parse the numbered list of suggestions
            suggestions = []
            for line in suggestions_text.strip().split('\n'):
                # Remove leading numbers, dots, and whitespace
                line = line.strip()
                if line and any(line.startswith(f"{i}.") or line.startswith(f"{i})") for i in range(1, 10)):
                    line = line[line.find(' ')+1:].strip()
                    suggestions.append(line)
            
            # If parsing failed or returned empty, use backup method
            if not suggestions:
                suggestions = [s.strip() for s in suggestions_text.split('\n') if s.strip()]
                suggestions = suggestions[:5]  # Limit to 5 suggestions
            
            # Update current suggestions
            self.current_suggestions = suggestions
            return suggestions
            
        except Exception as e:
            print(f"Error generating suggestions: {str(e)}")
            # If error occurs, keep existing suggestions
            return self.current_suggestions
    
    def update_llm(self, new_llm):
        """Update the LLM being used by the chatbot."""
        self.llm = new_llm
        # Reset chat history when model changes
        self.chat_history = []
        # Reset to initial suggestions
        self.current_suggestions = self.initial_suggestions.copy()
        return self.current_suggestions
    
    def update_temperature(self, temperature):
        """Update the temperature of the LLM."""
        self.llm.update_temperature(temperature)


def create_chatbot_interface():
    """Create and launch the Gradio interface for the chatbot."""
    
    # Define available models
    available_models = {
        "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
        "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1"
    }
    
    # Define demo mode and API key status
    use_real_api = False
    api_token = os.environ.get("HF_API_TOKEN", "")
    if api_token:
        use_real_api = True
    
    # Initialize with default model and temperature
    default_model = "Mistral-7B-Instruct-v0.2"
    default_temperature = 0.7
    model_name = available_models[default_model]
    
    # Create LLM based on API availability
    if use_real_api:
        llm = MistralLLM(model_name=model_name, api_token=api_token, temperature=default_temperature)
    else:
        llm = MockLLM(model_name=model_name, temperature=default_temperature)
    
    # Initialize chatbot with the LLM
    chatbot = ChatbotWithSuggestions(llm)
    
    def change_model(model_choice, temperature, api_key_input):
        """Handle model change, temperature update, and API key updates."""
        model_name = available_models[model_choice]
        
        # Validate temperature
        temp = float(temperature)
        temp = max(0.1, min(1.0, temp))  # Ensure temperature is between 0.1 and 1.0
        
        # Check if API key is provided
        use_api = False
        if api_key_input:
            use_api = True
            # Update environment variable for future uses
            os.environ["HF_API_TOKEN"] = api_key_input
        elif os.environ.get("HF_API_TOKEN"):
            use_api = True
        
        # Create appropriate LLM
        if use_api:
            new_llm = MistralLLM(
                model_name=model_name, 
                api_token=api_key_input or os.environ.get("HF_API_TOKEN"),
                temperature=temp
            )
            model_status = f"Using real API with model: {model_name} (Temperature: {temp})"
        else:
            new_llm = MockLLM(model_name=model_name, temperature=temp)
            model_status = f"Using mock model (demo mode): {model_name} (Temperature: {temp})"
        
        # Update chatbot with new LLM
        suggestions = chatbot.update_llm(new_llm)
        
        # Update UI components
        return model_status, [], "", suggestions[0], suggestions[1], suggestions[2], suggestions[3], suggestions[4]
    
    def update_temperature(temperature):
        """Update just the temperature without changing the model."""
        # Validate temperature
        temp = float(temperature)
        temp = max(0.1, min(1.0, temp))  # Ensure temperature is between 0.1 and 1.0
        
        # Update the temperature in the current LLM
        chatbot.update_temperature(temp)
        
        # Update model status display
        if isinstance(chatbot.llm, MistralLLM):
            model_status = f"Using real API with model: {chatbot.llm.model_name} (Temperature: {temp})"
        else:
            model_status = f"Using mock model (demo mode): {chatbot.llm.model_name} (Temperature: {temp})"
        
        return model_status
    
    def user_input_callback(message, history):
        """Process user input and update suggestions."""
        # Generate response
        response = chatbot.generate_response(message)
        
        # Generate new suggestions based on this interaction
        new_suggestions = chatbot.generate_suggestions(message, response)
        
        # Return response and update suggestion buttons
        return response, gr.Button.update(value=new_suggestions[0] if len(new_suggestions) > 0 else ""),\
               gr.Button.update(value=new_suggestions[1] if len(new_suggestions) > 1 else ""),\
               gr.Button.update(value=new_suggestions[2] if len(new_suggestions) > 2 else ""),\
               gr.Button.update(value=new_suggestions[3] if len(new_suggestions) > 3 else ""),\
               gr.Button.update(value=new_suggestions[4] if len(new_suggestions) > 4 else "")
    
    def suggestion_clicked(suggestion, history):
        """Handle when a suggestion button is clicked."""
        return suggestion, history + [[suggestion, None]]
    
    # Create the Gradio interface
    with gr.Blocks(css="#chatbot {height: 400px; overflow: auto;}") as interface:
        gr.Markdown("# AI Assistant with Model Selection and Dynamic Suggested Questions")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## Model Configuration")
                model_dropdown = gr.Dropdown(
                    choices=list(available_models.keys()),
                    value=default_model,
                    label="Select Model"
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=default_temperature,
                    step=0.1,
                    label="Temperature (0.1: Focused, 1.0: Creative)"
                )
                
                api_key_input = gr.Textbox(
                    placeholder="Enter HuggingFace API Token (optional)",
                    label="API Token",
                    type="password"
                )
                
                model_status = gr.Markdown(
                    f"Current mode: {'API' if use_real_api else 'Mock (Demo)'} (Temperature: {default_temperature})"
                )
                
                with gr.Row():
                    apply_temp_button = gr.Button("Update Temperature")
                    apply_model_button = gr.Button("Change Model", variant="primary")
            
            with gr.Column(scale=3):
                gr.Markdown("## Chat Interface")
                chatbot_component = gr.Chatbot(elem_id="chatbot")
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    lines=2,
                    label="Your Message"
                )
                
                with gr.Row():
                    clear = gr.Button("Clear Chat")
                    send = gr.Button("Send", variant="primary")
        
        gr.Markdown("### Suggested Questions")
        with gr.Row():
            suggestion_btn1 = gr.Button(chatbot.initial_suggestions[0])
            suggestion_btn2 = gr.Button(chatbot.initial_suggestions[1])
        
        with gr.Row():
            suggestion_btn3 = gr.Button(chatbot.initial_suggestions[2])
            suggestion_btn4 = gr.Button(chatbot.initial_suggestions[3])
            suggestion_btn5 = gr.Button(chatbot.initial_suggestions[4])
        
        # Set up event handlers
        send.click(user_input_callback, 
                  [msg, chatbot_component], 
                  [chatbot_component, suggestion_btn1, suggestion_btn2, suggestion_btn3, suggestion_btn4, suggestion_btn5])
        
        msg.submit(user_input_callback, 
                  [msg, chatbot_component], 
                  [chatbot_component, suggestion_btn1, suggestion_btn2, suggestion_btn3, suggestion_btn4, suggestion_btn5])
        
        clear.click(lambda: ([], [], chatbot.initial_suggestions[0], 
                          chatbot.initial_suggestions[1], 
                          chatbot.initial_suggestions[2],
                          chatbot.initial_suggestions[3],
                          chatbot.initial_suggestions[4]), 
                  None, 
                  [chatbot_component, msg, suggestion_btn1, suggestion_btn2, suggestion_btn3, suggestion_btn4, suggestion_btn5],
                  queue=False)
        
        # Apply model settings
        apply_model_button.click(change_model,
                         [model_dropdown, temperature_slider, api_key_input],
                         [model_status, chatbot_component, msg, suggestion_btn1, suggestion_btn2, suggestion_btn3, suggestion_btn4, suggestion_btn5])
        
        # Apply temperature update only
        apply_temp_button.click(update_temperature,
                         [temperature_slider],
                         [model_status])
        
        # Handle suggestion button clicks
        for btn in [suggestion_btn1, suggestion_btn2, suggestion_btn3, suggestion_btn4, suggestion_btn5]:
            btn.click(suggestion_clicked, 
                     [btn, chatbot_component], 
                     [msg, chatbot_component])
            
            # After clicking a suggestion, process it as if the user sent it
            btn.click(user_input_callback, 
                     [btn, chatbot_component], 
                     [chatbot_component, suggestion_btn1, suggestion_btn2, suggestion_btn3, suggestion_btn4, suggestion_btn5])
    
    return interface

# Launch the application
if __name__ == "__main__":
    interface = create_chatbot_interface()
    interface.launch()

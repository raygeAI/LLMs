# -*- coding: utf-8 -*-
import gradio as gr
import ollama
from typing import List, Tuple

model_name_dict = {
    "qwen2.5:32b": "qwen2.5:32b",
    "qwen2.5:72b": "qwen2.5:72b",
    "qwen2.5-coder": "qwen2.5-coder:32b",
    "qwq-32b": "qwq:32b-preview-q8_0",
}


# LLMAsServer 模型即服务， 启动聊天界面
class LLMAsServer:
    def __init__(self, model_name: str = model_name_dict["qwen2.5:32b"], is_stream=False):
        self.model_name = model_name
        self.is_stream = is_stream

    def generate(self, text: str):
        stream = ollama.generate(
            stream=self.is_stream,
            model=self.model_name,
            prompt=text,
        )
        response = ""
        if self.is_stream:
            for chunk in stream:
                response += chunk["response"]
        else:
            response = stream["response"]
        return response

    # api_generate 流式输出
    def _api_generate(self, chat_history: List):
        text = chat_history[-1]["content"]
        response = self.generate(text)
        yield chat_history + [{"role": "assistant", "content": response}]

    @staticmethod
    def _handle_user_message(user_message: str, history: list):
        """Handle the user submitted message. Clear message box and append to the history."""
        new_history = history + [{"role": "user", "content": user_message}]
        return '', new_history

    @staticmethod
    def _reset_chat() -> tuple:
        """Reset the agent's chat history. And clear all dialogue boxes."""
        return "", []

    def run(self):
        custom_css = """
            #box {
                height: 500px;
                overflow-y: scroll !important;
            }
            #clear-button {
                background-color: blue !important;
                color: white !important; /* Optional: Set text color to white for better contrast */
                border: none !important; /* Optional: Remove default border */
                border-radius: 2px !important; /* Optional: Add rounded corners */
                height: 150px !important; /* Set the height of the Textbox */
            }
            #message-input {
                width: 100% !important; /* Make the Textbox take up the full available width */
                max-width: calc(100% - 50px) !important; /* Adjust for the ClearButton's width */
                height: 150px !important; /* Increase the height to 50px */
            }
            #submit-button {
                background-color: green !important;
            }
            textarea{
                height: 100px !important; /* Ensure the inner container takes up the full height */
                border-radius: 5px !important; /* Optional: Add rounded corners */
                border-color: red !important; 
                border-style: solid !important; 
            }
            
        """
        demo = gr.Blocks(
            theme=gr.themes.Default(
                primary_hue="red",
                secondary_hue="pink"),
            # css="#box { height: 420px; overflow-y: scroll !important}",
            css=custom_css,
        )
        with demo:
            gr.Markdown(
                "# AI Chat \n"
                "### This Application  is Powered by The LLM Model {} \n".format(self.model_name.upper()),
            )
            chat_window = gr.Chatbot(
                label="Message History",
                scale=5,
                type='messages',
                elem_id="chatbot-container"
            )
            with gr.Row():
                message_input = gr.Textbox(label="Input", placeholder="Type your message here...", scale=5, elem_id="message-input")
                clear = gr.ClearButton(elem_id="clear-button")
            submit_button = gr.Button("Submit", elem_id="submit-button")

            submit_button.click(
                fn=self._handle_user_message,
                inputs=[message_input, chat_window],
                outputs=[message_input, chat_window]
            ).then(
                fn=self._api_generate,
                inputs=chat_window,
                outputs=chat_window
            )

            message_input.submit(
                self._handle_user_message,
                inputs=[message_input, chat_window],
                outputs=[message_input, chat_window],
            ).then(
                self._api_generate,
                chat_window,
                chat_window,
            )
            clear.click(self._reset_chat, None, [message_input, chat_window])

        demo.launch(show_error=True, share=False, server_port=7861)


if __name__ == "__main__":
    server = LLMAsServer(model_name=model_name_dict["qwen2.5-coder"])
    server.run()

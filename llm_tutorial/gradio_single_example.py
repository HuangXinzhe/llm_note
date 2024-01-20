"""
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 50072 --model /root/autodl-fs/model/Llama-2-7b-hf --served-model-name llama-2-7b-hf
"""

import gradio as gr
import requests

models = ['llama-2-7b-hf']


def completion(question):
    model_url_dict = {models[0]: "http://localhost:50072/v1/chat/completions",
                      }
    answers = []
    for model in models:
        headers = {'Content-Type': 'application/json'}

        json_data = {
            'model': model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant.'
                },
                {
                    'role': 'user',
                    'content': question
                },
            ],
        }

        response = requests.post(model_url_dict[model], headers=headers, json=json_data)
        answer = response.json()["choices"][0]["message"]["content"]
        answers.append(answer)
    return answers


demo = gr.Interface(
    fn=completion,
    inputs=gr.Textbox(lines=5, placeholder="input your question", label="question"),
    outputs=[gr.Textbox(lines=5, placeholder="answer", label=models[0])]
)

demo.launch(server_name='0.0.0.0', share=True)
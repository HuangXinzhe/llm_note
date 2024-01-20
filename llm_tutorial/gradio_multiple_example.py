"""
CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 50072 --model /data-ai/model/llama2/llama2_hf/Llama-2-13b-chat-hf --served-model-name llama-2-13b-chat-hf
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 50073 --model /data-ai/model/baichuan2/Baichuan2-13B-Chat --served-model-name Baichuan2-13B-Chat --trust-remote-code --chat-template /data-ai/usr/code/template_baichuan.jinja
"""


import gradio as gr
import requests

models = ['llama-2-13b-chat-hf', 'Baichuan2-13B-Chat']


def completion(question):
    model_url_dict = {models[0]: "http://localhost:50072/v1/chat/completions",
                      models[1]: "http://localhost:50073/v1/chat/completions",
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
    outputs=[gr.Textbox(lines=5, placeholder="answer", label=models[0]),
             gr.Textbox(lines=5, placeholder="answer", label=models[1])]
)

demo.launch(server_name='0.0.0.0', share=True)
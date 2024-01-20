"""
1、线下批量推理
"""
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="/root/autodl-fs/model/Llama-2-7b-hf")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


"""
2、API服务启动命令:
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.api_server --model /content/drive/MyDrive/llm/model/TinyLlama/TinyLlama-1.1B-Chat-v1.0
python -m vllm.entrypoints.api_server --model /root/autodl-fs/model/Llama-2-7b-hf
"""

"""
3、OpenAI风格的API服务
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server --model /data-ai/model/llama2/llama2_hf/Llama-2-13b-chat-hf --served-model-name llama-2-13b-chat-hf

查看模型
curl http://localhost:8000/v1/models

text completion
代码见text_completion.py
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'

chat completion
代码见chat_completion.py
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
"""
from transformers import AutoModelForCausalLM
from config import MODEL_NAME, MODEL_PATH
from transformers import AutoTokenizer

# 定义模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True)

# 节省显存
model.gradient_checkpointing_enable()

# 定义tokenizer
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token_id = 0

named_labels = ['neg', 'pos']

label_ids = [
    tokenizer(named_labels[i], add_special_tokens=False)["input_ids"][0]
    for i in range(len(named_labels))
]

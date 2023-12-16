MAX_LEN = 32             # 最大长度
SEED = 42                 # 随机种子
LR = 2e-5                 # 学习率
BATCH_SIZE = 8          # BATCH_SIZE
WARMUP_RATIO = 0.1        # warmup比例
INTERVAL = 100             # 每多少步打一次 log / 做一次 eval

MODEL_NAME = "gpt2"                     # 模型名称
# MODEL_NAME = "gpt2-large"
MODEL_PATH = "/Volumes/WD_BLACK/models/gpt2/"  # 模型路径

DATASET_NAME = "rotten_tomatoes"        # 数据集名称
DATA_BODY_KEY = "text"
DATA_LABEL_KEY = "label"

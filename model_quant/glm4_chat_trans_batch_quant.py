"""
使用 batch 基于 glm-4-9b-chat 量化版推理示例。

基于 batch_message，笔者使用的 ubuntu 22.04、4090显卡，代码运行时显存占用 8156MiB。

Notes: 如果 batch_message 较大，显存招用会明显增加。

Requirements:
pip install bitsandbytes
pip install 'accelerate>=0.26.0'
"""

from typing import Optional, Union
from transformers import AutoModel, AutoTokenizer, LogitsProcessorList, BitsAndBytesConfig

MODEL_PATH = '/root/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat/'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,   # 加载模型为 INT8
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    encode_special_tokens=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, quantization_config=bnb_config, device_map="auto").eval()


def process_model_outputs(inputs, outputs, tokenizer):
    responses = []
    for input_ids, output_ids in zip(inputs.input_ids, outputs):
        response = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True).strip()
        responses.append(response)
    return responses


def batch(
        model,
        tokenizer,
        messages: Union[str, list[str]],
        max_input_tokens: int = 8192,
        max_new_tokens: int = 8192,
        num_beams: int = 1,
        do_sample: bool = True,
        top_p: float = 0.8,
        temperature: float = 0.8,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
):
    messages = [messages] if isinstance(messages, str) else messages
    batched_inputs = tokenizer(messages, return_tensors="pt", padding="max_length", truncation=True,
                               max_length=max_input_tokens).to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": do_sample,
        "top_p": top_p,
        "temperature": temperature,
        "logits_processor": logits_processor,
        "eos_token_id": model.config.eos_token_id
    }
    batched_outputs = model.generate(**batched_inputs, **gen_kwargs)
    batched_response = process_model_outputs(batched_inputs, batched_outputs, tokenizer)
    return batched_response


if __name__ == "__main__":

    batch_message = [
        [
            {"role": "user", "content": "我的爸爸和妈妈结婚为什么不能带我去"},
            {"role": "assistant", "content": "因为他们结婚时你还没有出生"},
            {"role": "user", "content": "我刚才的提问是"}
        ],
        [
            {"role": "user", "content": "你好，你是谁"}
        ]
    ]

    batch_inputs = []
    max_input_tokens = 1024
    for i, messages in enumerate(batch_message):
        new_batch_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        max_input_tokens = max(max_input_tokens, len(new_batch_input))
        batch_inputs.append(new_batch_input)
    gen_kwargs = {
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": 8192,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.8,
        "num_beams": 1,
    }

    batch_responses = batch(model, tokenizer, batch_inputs, **gen_kwargs)
    for response in batch_responses:
        print("=" * 10)
        print(response)
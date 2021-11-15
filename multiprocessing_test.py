from multiprocessing import Process
from transformers import GPT2Tokenizer, FlaxGPTNeoModel


tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
model = FlaxGPTNeoModel.from_pretrained('EleutherAI/gpt-neo-1.3B')
inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')


def froward_func():
    print(f"Model: {model}")
    print(f"Input: {inputs}")
    output = model(**inputs)
    print(f'Output: {output}')
    return output


print('Single process:')
froward_func()

print('\n\nMulti process:')
forward_process = Process(target=froward_func)
forward_process.start()
forward_process.join(timeout=60)

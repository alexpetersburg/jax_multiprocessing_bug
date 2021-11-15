from multiprocessing import Process
from transformers import GPT2Tokenizer, FlaxGPTNeoModel


tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
model = FlaxGPTNeoModel.from_pretrained('EleutherAI/gpt-neo-1.3B')
inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')


def froward_func():
    print("starting func")
    output = model(**inputs)
    print(output)
    return output


print(f'Single process:')
froward_func()

print(f'Multi process:')
forward_process = Process(target=froward_func)
forward_process.start()
forward_process.join(timeout=60)

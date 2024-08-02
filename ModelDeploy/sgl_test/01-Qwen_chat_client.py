"""
docker run --gpus "device=1"    -p 30000:30000     -v /data/bocheng/data/.cache/huggingface:/root/.cache/huggingface     --ipc=host     lmsysorg/sglang:latest     python3 -m sglang.launch_server --model-path Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4 --host 0.0.0.0 --port 30000
"""

from sglang import (
    function,
    system,
    user,
    assistant,
    gen,
    set_default_backend,
    RuntimeEndpoint,
)


@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))


set_default_backend(RuntimeEndpoint("http://192.168.1.14:30000"))

state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
)

for m in state.messages():
    print(m["role"], ":", m["content"])

print(state["answer_1"])

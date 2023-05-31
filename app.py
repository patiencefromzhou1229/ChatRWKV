import gradio as gr
import os, gc, copy, torch
from datetime import datetime
from huggingface_hub import hf_hub_download
from pynvml import *

nvmlInit()
gpu_h = nvmlDeviceGetHandleByIndex(0)
ctx_limit = 1536
title = "RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192"

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'  # if '1' then use CUDA kernel for seq mode (much faster)

from rwkv.model import RWKV

# model_path = hf_hub_download(repo_id="BlinkDL/rwkv-4-raven", filename=f"{title}.pth")
model_path = '/app/models/{}.pth'.format(title)
model = RWKV(model=model_path, strategy='cuda fp16i8 *24 -> cuda fp16')
from rwkv.utils import PIPELINE, PIPELINE_ARGS

pipeline = PIPELINE(model, "20B_tokenizer.json")


def generate_prompt(instruction, input=None):
    instruction = instruction.strip().replace('\r\n', '\n').replace('\n\n', '\n')
    input = input.strip().replace('\r\n', '\n').replace('\n\n', '\n')
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
# Instruction:
{instruction}
# Input:
{input}
# Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
# Instruction:
{instruction}
# Response:
"""


def evaluate(
        instruction,
        input=None,
        token_count=200,
        temperature=1.0,
        top_p=0.7,
        presencePenalty=0.1,
        countPenalty=0.1,
):
    args = PIPELINE_ARGS(temperature=max(0.2, float(temperature)), top_p=float(top_p),
                         alpha_frequency=countPenalty,
                         alpha_presence=presencePenalty,
                         token_ban=[],  # ban the generation of some tokens
                         token_stop=[0])  # stop generation whenever you see any token here

    instruction = instruction.strip().replace('\r\n', '\n').replace('\n\n', '\n')
    input = input.strip().replace('\r\n', '\n').replace('\n\n', '\n')
    ctx = generate_prompt(instruction, input)

    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    for i in range(int(token_count)):
        out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield out_str.strip()
            out_last = i + 1

    gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    print(f'vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')
    del out
    del state
    gc.collect()
    torch.cuda.empty_cache()
    yield out_str.strip()


examples = [
    ["Tell me about ravens.", "", 300, 1.2, 0.5, 0.4, 0.4],
    ["Write a python function to mine 1 BTC, with details and comments.", "", 300, 1.2, 0.5, 0.4, 0.4],
    ["Write a song about ravens.", "", 300, 1.2, 0.5, 0.4, 0.4],
    ["Explain the following metaphor: Life is like cats.", "", 300, 1.2, 0.5, 0.4, 0.4],
    ["Write a story using the following information", "A man named Alex chops a tree down", 300, 1.2, 0.5, 0.4, 0.4],
    ["Generate a list of adjectives that describe a person as brave.", "", 300, 1.2, 0.5, 0.4, 0.4],
    [
        "You have $100, and your goal is to turn that into as much money as possible with AI and Machine Learning. Please respond with detailed plan.",
        "", 300, 1.2, 0.5, 0.4, 0.4],
]

##########################################################################

chat_intro = '''The following is a coherent verbose detailed conversation between <|user|> and an AI girl named <|bot|>.
<|user|>: Hi <|bot|>, Would you like to chat with me for a while?
<|bot|>: Hi <|user|>. Sure. What would you like to talk about? I'm listening.
'''


def user(message, chatbot):
    chatbot = chatbot or []
    # print(f"User: {message}")
    return "", chatbot + [[message, None]]


def alternative(chatbot, history):
    if not chatbot or not history:
        return chatbot, history

    chatbot[-1][1] = None
    history[0] = copy.deepcopy(history[1])

    return chatbot, history


def chat(
        prompt,
        user,
        bot,
        chatbot,
        history,
        temperature=1.0,
        top_p=0.8,
        presence_penalty=0.1,
        count_penalty=0.1,
):
    args = PIPELINE_ARGS(temperature=max(0.2, float(temperature)), top_p=float(top_p),
                         alpha_frequency=float(count_penalty),
                         alpha_presence=float(presence_penalty),
                         token_ban=[],  # ban the generation of some tokens
                         token_stop=[])  # stop generation whenever you see any token here

    if not chatbot:
        return chatbot, history

    message = chatbot[-1][0]
    message = message.strip().replace('\r\n', '\n').replace('\n\n', '\n')
    ctx = f"{user}: {message}\n\n{bot}:"

    if not history:
        prompt = prompt.replace("<|user|>", user.strip())
        prompt = prompt.replace("<|bot|>", bot.strip())
        prompt = prompt.strip()
        prompt = f"\n{prompt}\n\n"

        out, state = model.forward(pipeline.encode(prompt), None)
        history = [state, None, []]  # [state, state_pre, tokens]
        # print("History reloaded.")

    [state, _, all_tokens] = history
    state_pre_0 = copy.deepcopy(state)

    out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:], state)
    state_pre_1 = copy.deepcopy(state)  # For recovery

    # print("Bot:", end='')

    begin = len(all_tokens)
    out_last = begin
    out_str: str = ''
    occurrence = {}
    for i in range(300):
        if i <= 0:
            nl_bias = -float('inf')
        elif i <= 30:
            nl_bias = (i - 30) * 0.1
        elif i <= 130:
            nl_bias = 0
        else:
            nl_bias = (i - 130) * 0.25
        out[187] += nl_bias
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        next_tokens = [token]
        if token == 0:
            next_tokens = pipeline.encode('\n\n')
        all_tokens += next_tokens

        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        out, state = model.forward(next_tokens, state)

        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            # print(tmp, end='', flush=True)
            out_last = begin + i + 1
            out_str += tmp

            chatbot[-1][1] = out_str.strip()
            history = [state, all_tokens]
            yield chatbot, history

        out_str = pipeline.decode(all_tokens[begin:])
        out_str = out_str.replace("\r\n", '\n').replace('\\n', '\n')

        if '\n\n' in out_str:
            break

        # State recovery
        if f'{user}:' in out_str or f'{bot}:' in out_str:
            idx_user = out_str.find(f'{user}:')
            idx_user = len(out_str) if idx_user == -1 else idx_user
            idx_bot = out_str.find(f'{bot}:')
            idx_bot = len(out_str) if idx_bot == -1 else idx_bot
            idx = min(idx_user, idx_bot)

            if idx < len(out_str):
                out_str = f" {out_str[:idx].strip()}\n\n"
                tokens = pipeline.encode(out_str)

                all_tokens = all_tokens[:begin] + tokens
                out, state = model.forward(tokens, state_pre_1)
                break

    gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    print(f'vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')

    gc.collect()
    torch.cuda.empty_cache()

    chatbot[-1][1] = out_str.strip()
    history = [state, state_pre_0, all_tokens]
    yield chatbot, history


##########################################################################

with gr.Blocks(title=title) as demo:
    gr.HTML(f"<div style=\"text-align: center;\">\n<h1>🐦Raven - {title}</h1>\n</div>")
    with gr.Tab("Instruct mode"):
        gr.Markdown(
            f"Raven is [RWKV 14B](https://github.com/BlinkDL/ChatRWKV) 100% RNN [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) finetuned to follow instructions. *** Please try examples first (bottom of page) *** (edit them to use your question). Demo limited to ctxlen {ctx_limit}. Finetuned on alpaca, gpt4all, codealpaca and more. For best results, *** keep you prompt short and clear ***. <b>UPDATE: now with Chat (see above, as a tab) ==> turn off as of now due to VRAM leak caused by buggy code.</b>.")
        with gr.Row():
            with gr.Column():
                instruction = gr.Textbox(lines=2, label="Instruction", value="Tell me about ravens.")
                input = gr.Textbox(lines=2, label="Input", placeholder="none")
                token_count = gr.Slider(10, 300, label="Max Tokens", step=10, value=300)
                temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.2)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.5)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0.4)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=0.4)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=5)
        data = gr.Dataset(
            components=[instruction, input, token_count, temperature, top_p, presence_penalty, count_penalty],
            samples=examples, label="Example Instructions",
            headers=["Instruction", "Input", "Max Tokens", "Temperature", "Top P", "Presence Penalty", "Count Penalty"])
        submit.click(evaluate, [instruction, input, token_count, temperature, top_p, presence_penalty, count_penalty],
                     [output])
        clear.click(lambda: None, [], [output])
        data.click(lambda x: x, [data],
                   [instruction, input, token_count, temperature, top_p, presence_penalty, count_penalty])

    # with gr.Tab("Chat (Experimental - Might be buggy - use ChatRWKV for reference)"):
    #     gr.Markdown(f'''<b>*** The length of response is restricted in this demo. Use ChatRWKV for longer generations. ***</b> Say "go on" or "continue" can sometimes continue the response. If you'd like to edit the scenario, make sure to follow the exact same format: empty lines between (and only between) different speakers. Changes only take effect after you press [Clear]. <b>The default "Bob" & "Alice" names work the best.</b>''', label="Description")
    #     with gr.Row():
    #         with gr.Column():
    #             chatbot = gr.Chatbot()
    #             state = gr.State()
    #             message = gr.Textbox(label="Message", value="Write me a python code to land on moon.")
    #             with gr.Row():
    #                 send = gr.Button("Send", variant="primary")
    #                 alt = gr.Button("Alternative", variant="secondary")
    #                 clear = gr.Button("Clear", variant="secondary")
    #         with gr.Column():
    #             with gr.Row():
    #                 user_name = gr.Textbox(lines=1, max_lines=1, label="User Name", value="Bob")
    #                 bot_name = gr.Textbox(lines=1, max_lines=1, label="Bot Name", value="Alice")
    #             prompt = gr.Textbox(lines=10, max_lines=50, label="Scenario", value=chat_intro)
    #             temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.2)
    #             top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.5)
    #             presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=0.4)
    #             count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=0.4)
    #     chat_inputs = [
    #         prompt,
    #         user_name,
    #         bot_name,
    #         chatbot,
    #         state,
    #         temperature,
    #         top_p,
    #         presence_penalty,
    #         count_penalty
    #     ]
    #     chat_outputs = [chatbot, state]
    #     message.submit(user, [message, chatbot], [message, chatbot], queue=False).then(chat, chat_inputs, chat_outputs)
    #     send.click(user, [message, chatbot], [message, chatbot], queue=False).then(chat, chat_inputs, chat_outputs)
    #     alt.click(alternative, [chatbot, state], [chatbot, state], queue=False).then(chat, chat_inputs, chat_outputs)
    #     clear.click(lambda: ([], None, ""), [], [chatbot, state, message], queue=False)

demo.queue(concurrency_count=1, max_size=10)
demo.launch(share=True)
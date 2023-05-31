import gradio as gr
import os, gc, torch
import win32ui
from datetime import datetime
#from huggingface_hub import hf_hub_download
#from pynvml import *

#nvmlInit()
#gpu_h = nvmlDeviceGetHandleByIndex(0)
ctx_limit = 2048
desc = f'''链接：
<a href='https://github.com/BlinkDL/ChatRWKV' target="_blank" style="margin:0 0.5em">ChatRWKV</a>
<a href='https://github.com/BlinkDL/RWKV-LM' target="_blank" style="margin:0 0.5em">RWKV-LM</a>
<a href="https://pypi.org/project/rwkv/" target="_blank" style="margin:0 0.5em">RWKV pip package</a>
<a href="https://zhuanlan.zhihu.com/p/609154637" target="_blank" style="margin:0 0.5em">知乎教程</a>
<a href="https://zhuanlan.zhihu.com/p/616815736" target="_blank" style="margin:0 0.5em">webui教程</a>
'''

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'  # if '1' then use CUDA kernel for seq mode (much faster)

from rwkv.model import RWKV
dir_path = os.path.dirname(os.path.realpath(__file__))
# 选择模型


# 0代表另存为对话框，1代表打开文件对话框
dlg = win32ui.CreateFileDialog(1)

# 设置默认目录
dlg.SetOFNInitialDir('D:/chatRWKV/ChatRWKV/models/')

# 显示对话框
dlg.DoModal()

# 获取用户选择的文件全路径
filename = dlg.GetPathName()

model_path = filename


#model_path = os.path.join(dir_path, 'models','RWKV-4-Pile-7B-EngChn-testNovel-done-ctx2048-20230317')
#model_path = "D:/chatRWKV/ChatRWKV/models/RWKV-4-Pile-7B-EngChn-testNovel-done-ctx2048-20230317"
model = RWKV(model=model_path, strategy='cuda fp16i8')
from rwkv.utils import PIPELINE, PIPELINE_ARGS
relative_path = os.path.join(dir_path, 'v2', '20B_tokenizer.json')
pipeline = PIPELINE(model, relative_path)


def infer(
        ctx,
        preprocess,  # 新增一个参数，指示是否进行预处理
        token_count=10,
        temperature=1.0,
        top_p=0.8,
        presencePenalty=0.1,
        countPenalty=0.1,
):

    args = PIPELINE_ARGS(temperature=max(0.2, float(temperature)), top_p=float(top_p),
                         alpha_frequency=countPenalty,
                         alpha_presence=presencePenalty,
                         token_ban=[0],  # ban the generation of some tokens
                         token_stop=[])  # stop generation whenever you see any token here

    if preprocess == "问答模式":
        ctx = f'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.
# Instruction:
{ctx.strip()}
# Response:
'''
    elif preprocess == "续写模式":
        ctx = ctx.strip().split('\n')
        for c in range(len(ctx)):
            ctx[c] = ctx[c].strip().strip('\u3000').strip('\r')
        ctx = list(filter(lambda c: c != '', ctx))
        ctx = '\n' + ('\n'.join(ctx)).strip()
    else:
        ctx = ctx.strip()
    # gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    # print(f'vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')

    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    for i in range(int(token_count)):
        out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
        for n in args.token_ban:
            out[n] = -float('inf')
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
            yield out_str
            out_last = i + 1
    gc.collect()
    torch.cuda.empty_cache()
    yield out_str
# 上面这段代码实现了生成文本的功能，具体可以分为以下几个步骤：
# 构造参数对象args。这里使用PIPELINE_ARGS函数构造参数对象，其中包括了生成文本的一些参数，如温度、top-p采样等，以及一些额外控制因素，如存在惩罚和出现次数惩罚等。
# 对输入的上下文进行预处理。首先将输入的字符串按行划分成多个文本行，并去除每行首尾的空格、中文空格和回车符等无关字符，然后将这些文本行连接起来并在开头添加一个换行符，得到上下文字符串ctx。
# 初始化变量。将所有token放入列表all_tokens中，初始化输出字符串out_str为空字符串，记录当前已经输出的token序列的最后一个位置out_last，记录每个token的出现次数的字典occurrence，以及模型的初始状态state（默认为None）。
# 循环生成指定数量的token。在循环中，先使用model.forward函数生成下一个token的概率分布out和下一个时间步的状态state。在概率分布中对禁止出现的token进行特殊处理，对已经出现过的token进行惩罚。
# 然后调用pipeline.sample_logits函数基于概率分布采样下一个token，并将其添加到all_tokens中。最后检查该token是否在停止词列表中，如果是，则终止循环。
# 将生成的token列表转换回文本字符串。每次生成新的token后，都将当前所有token列表（从上一次输出结束位置开始）解码为文本，并将新的文本与之前的输出字符串拼接起来。对于没有解码成功的token，暂不添加到最终结果中，等待后续token的补充。
# 输出结果。每次解码成功新的一段文本后，都将其通过yield关键字输出，即将文本段作为一个生成器的元素返回给调用者。这样做的好处是可以在生成文本的过程中实时地输出部分结果，而不是等待整个生成过程结束后再输出。
# 清理内存。在循环结束后，进行内存清理操作，包括清空无用对象和释放GPU内存等。最后再次使用yield输出最终的结果。

examples = [
    ["以下是不朽的科幻史诗巨著，描写细腻，刻画了宏大的星际文明战争。\n第一章","续写模式", 200, 1, 0.8, 0.1, 0.1],
    ["“区区","续写模式", 200, 1, 0.8, 0.1, 0.1],
    ["这是一个玄幻修真世界，详细世界设定如下：\n1.","续写模式", 200, 1, 0.8, 0.1, 0.1],
    ["这是一个奇幻魔法世界，详细世界设定如下：\n1.","续写模式", 200, 1, 0.8, 0.1, 0.1],
    ["怎样创立一家快速盈利的AI公司：\n1.","续写模式", 200, 1, 0.8, 0.1, 0.1],
    ["帝皇是一名极为强大的灵能者，而且还是永生者：一个拥有无穷知识与力量以及使用它们的雄心的不朽存在。根据传说，","续写模式", 200, 1, 0.8, 0.1, 0.1],
    ["我问智脑：“三体人发来了信息，告诉我不要回答，这是他们的阴谋吗？”","续写模式", 200, 1, 0.8, 0.1, 0.1],
    ["我问编程之神：“Pytorch比Tensorflow更好用吗？”","续写模式", 200, 1, 0.8, 0.1, 0.1],
    ["Translation Samples\nChinese: 修道之人，最看重的是什么？\nEnglish:","续写模式", 200, 1, 0.5, 0.1, 0.1],
    ["import torch","续写模式", 200, 1, 0.8, 0.1, 0.1],
    ["“三体人的修仙功法与地球不同，最大的区别","续写模式", 200, 1, 0.8, 0.1, 0.1],
    ["“我们都知道，魔法的运用有四个阶段，第一","续写模式", 200, 1, 0.8, 0.1, 0.1],
    ["丹妮摸了摸，衣料柔软如水般流过她的手指，她从来","续写模式", 200, 1, 0.8, 0.1, 0.1],
    ["无论怎样，我必须将这些恐龙养大","续写模式", 200, 1, 0.8, 0.1, 0.1],
    ["通过基因改造，修真","续写模式", 200, 1, 0.8, 0.1, 0.1],
    ["开题报告","问答模式", 200, 1, 0.8, 0.1, 0.1],
    ["你好，你能做个自我介绍吗？","聊天模式", 100, 1, 0.8, 0.1, 0.1],
]

iface = gr.Interface(
    fn=infer,
    description=f'''<b>请点击例子（在页面底部）</b>，可编辑内容。这里只看输入的最后约1100字，请写好，标点规范，无错别字，否则电脑会模仿你的错误。<b>为避免占用资源，每次生成限制长度。可将输出内容复制到输入，然后继续生成</b>。推荐提高temp改善文采，降低topp改善逻辑，提高两个penalty避免重复，具体幅度请自己实验</b>。续写可以把输出重新复制到输入就可以继续续写。{desc}''',
    allow_flagging="never",
    inputs=[
        gr.Textbox(lines=10, label="prompt输入", value="以下是不朽的科幻史诗巨著，描写细腻，刻画了宏大的星际文明战争。\n第一章"),  # prompt
        gr.Radio(["问答模式", "续写模式"], label="模式选择", info="请选择一个模式"),# 新增一个控件
        gr.Slider(10, 1000, step=10, value=200, label="token_count 每次生成的长度"),  # token_count
        gr.Slider(0.2, 2.0, step=0.1, value=1, label="temperature 默认1，高则变化丰富，低则保守求稳"),  # temperature
        gr.Slider(0.0, 1.0, step=0.05, value=0.8, label="top_p 默认0.8，高则标新立异，低则循规蹈矩"),  # top_p
        gr.Slider(0.0, 1.0, step=0.1, value=0.1, label="presencePenalty 默认0.1，避免写过的类似字"),  # presencePenalty
        gr.Slider(0.0, 1.0, step=0.1, value=0.1, label="countPenalty 默认0.1，额外避免写过多次的类似字"),  # countPenalty

    ],
    outputs=gr.Textbox(label="输出", lines=28),
    # 这里的gr.Interface是Gradio提供的一个接口创建器，用于生成交互式界面。在代码中，我们通过调用该接口创建器，并传递自定义的函数、描述信息、输入输出控件等参数来创建一个新的Gradio接口。
    # 具体地，上述代码中的infer函数是我们自己编写的生成文本的函数，可以被Gradio调用以产生输出结果。而输入控件则包括一个多行文本框（gr.Textbox）用于输入上下文，
    # 以及一些滑动条（gr.Slider）用于控制生成文本的各种参数（如生成长度、温度、top-p等）。输出控件是一个另外的多行文本框，用于显示生成的续写结果。
    # 当用户在输入控件中输入一段文本时，Gradio将会将该文本作为参数调用infer函数，同时将其他用户设置的参数一起传递给该函数。在infer函数中，
    # 我们可以通过ctx参数获取到用户输入的上下文，然后利用该上下文进行文本生成，最终将生成的结果输出。因此，我们可以通过这个方式实现获取用户输入的上下文。
    examples=examples,
    cache_examples=False,
).queue()

demo = gr.TabbedInterface(
    [iface], ["chatRWKV-Webui"]
)

demo.queue(max_size=5)
demo.launch(share=False)
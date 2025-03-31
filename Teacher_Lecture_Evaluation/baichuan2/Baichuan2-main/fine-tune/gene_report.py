import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from peft import PeftModel
import networkx as nx
import os
import pandas as pd
import glob


model_path = 'baichuan2/Baichuan2-main/baichuan'

lora_path = '/pth/output'

def init_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()

    model.generation_config = GenerationConfig.from_pretrained(
        model_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def query_knowledge_graph(user_input, knowledge_graph):
    node_counts = {}
    for node in knowledge_graph.nodes():
        if node.lower() in user_input.lower():
            node_counts[node] = node_counts.get(node, 0) + 1
    top_nodes = sorted(node_counts, key=node_counts.get, reverse=True)[:3]
    topKey = f"Key: {', '.join(top_nodes)}"

    related_nodes = set(top_nodes)
    for node in top_nodes:
        neighbors = nx.single_source_shortest_path_length(knowledge_graph, node, cutoff=1).keys()
        related_nodes.update(neighbors)

    topKey += f"{', '.join(map(str, related_nodes))}"

    return topKey

EP = """
Question：你现在是一个教师教学语言评价专家，根据以下的四个方面(分别为教学环节、教学目标与内容、教学过程与方法、师生互动，每个方面分成四个等级，每个等级都有对应的分数)根据标准来评价教师教学语言，并且你的评价内容需要给出等级及相应等级所对应的分数(分数务必对应好是哪一个评价方面的，对应等级的对应分数也别弄错){
(1. 教学环节：等级A(分数:9~10分)：教学情境创设有新意，环节清晰，过渡自然。等级B(分数:7~8分)：教学情境创设较为新颖，环节基本清晰，过渡较为自然，但存在不足。等级C(分数:6分)：教学情境创设略显平淡，环节不够清晰，过渡有些生硬，存在一些问题。等级D(分数:0~5分)：教学情境创设平淡，环节混乱，过渡不自然，需要较大改进。)
(2. 教学目标与内容：等级A(分数:9~10分)：教学目标明确，重点突出，学习内容、任务与活动安排适宜，教学内容突出学科特色。等级B(分数:7~8分)：教学目标基本明确，重点有一定突出，学习内容、任务与活动安排基本适宜，但略有不足。等级C(分数:6分)：教学目标不够明确，重点不够突出，学习内容、任务与活动安排不够适宜，存在一些问题。等级D(分数:0~5分)：教学目标不明确，重点不突出，学习内容、任务与活动安排不适宜，需要重大改进。)
(3. 教学过程与方法：等级A(分数:9~10分)：教学方法有效针对所教内容，能照顾到学生的个别差异，注重激发学生学习兴趣与动机。等级B(分数:7~8分)：教学方法基本针对所教内容，对学生个别差异照顾较少，较重视学生学习兴趣与动机，但某些方面可能需要改进。等级C(分数:6分)：教学方法针对所教内容有些不足，对学生个别差异照顾不够，学生学习兴趣与动机不够重视，需要改进。等级D(分数:0~5分)：教学方法未针对所教内容，对学生个别差异照顾不到位，未重视学生学习兴趣与动机，存在明显不足，需要重大改进。)
(4. 师生互动：等级A(分数:9~10分)：能够利用很多提问促进学生的思考，鼓励并赞扬学生的优良表现，及时纠正学生的错误。等级B(分数:7~8分)：能够利用一定提问促进学生的思考，存在鼓励并赞扬学生的表现，但略有不足。等级C(分数:6分)：很少利用提问促进学生思考，鼓励与表扬学生不够充分，纠正学生错误不够及时，存在不足。等级D(分数:0~5分)：未能有效利用提问促进学生思考，未能鼓励与表扬学生，纠正学生错误不及时。需要重大改进。)
}请你记住这四个评价标准，并且你的评价内容根据评价标准需要给出等级及相应等级所对应的分数。
Answer：好的,我将会用这个评价标准来评价教师教学语言。
Question：下面会给你一个新的教师教学录音文本，根据这些评价标准来生成相应的内容。
Answer：请提供一份新的教师教学录音文本,我会根据上述模板的格式根据评教标准来生成教师教学语言评价报告。
Question：现在我给你一份新的教师教学录音文本，请把这个教师教学录音文本根据上述的评价指标及对应的模板格式，最后来生成一个教师教学语言评价报告，给出每个方面（教学环节、教学目标与内容、教学过程与方法、师生互动）对应等级和相应等级的分数及评价，最后给出一个总体评价和意见。下面是这个教师教学录音文本的内容：
"""

mb = """
Question:我给你一个教师教学语言评价报告模板，具体格式内容如下{
1.教学环节：等级：...，评价：...
2.教学目标与内容：等级：...，评价：...
3.教学过程与方法：等级：...，评价：...
4.师生互动：等级：...，评价：...
总体评价：...，
给出的建议：...}请你后面对一个新的教师教学录音文本评价时按照这个模板格式来写“...”表示你要根据评价指标对一份教师教学录音文本进行评价的等级对应分数和内容。
Answer:好的，我会根据这个模板生成教师教学语言评价报告。
"""



def init_chat_history():
    messages = []
    return messages

def generate_evaluation_report(content):
    model, tokenizer = init_model()
    # 遍历Word文档中的每个段落，并将其文本内容添加到列表中
    messages = init_chat_history()  # 初始化消息列表
    prompt = EP + content + """。以上是我完整的教师教学录音文本内容，请把这个教师教学录音文本根据上述的评价指标，最后来生成一个评价报告，给出每个方面（教学环节、教学目标与内容、教学过程与方法、师生互动）的等级和相应等级的分数(满分为10分)及评价，评价内容需要有知识点。"""
    knowledge_graph_file = "/home/amax/ghh/baichuan2/Baichuan2-main/knowledge_graph.graphml"
    knowledge_graph = nx.read_graphml(knowledge_graph_file)
    topKey = query_knowledge_graph(prompt, knowledge_graph)
    prompt = """Question:这份教师教学录音文本的知识点为："""+f"{topKey}" + """这是一些关键词，将这些关键词作为知识点，在生成的评价报告中一定要出现这些知识点.""" + """Answer：好的，我会在评价报告中加入这些知识点。   """ \
                 + """Answer:请提供一个模板来根据评价标准来评价这份教师教学录音文本。""" + mb + """question：我已知晓模板，请给出具体的评价标准。  """ + prompt + """Answer:"""

    messages.append({"role": "user", "content": prompt})
    evaluation_results = []

    for response in model.chat(tokenizer, messages, stream=True, generation_config=model.generation_config):
        evaluation_results.append(response)
    messages.append({"role": "assistant", "content": response})
    print(response)
    return update_evaluation(response)

def get_teaching_report(txt_file):
    with open(txt_file, "r") as file:
        content = file.read()
    return generate_evaluation_report(content)


if __name__ == '__main__':
    folder_path = "baichuan2/Baichuan2-main/fine-tune/input/"
    out_path = "baichuan2/Baichuan2-main/fine-tune/output/"
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    for txt_file in txt_files:
        file_name = os.path.splitext(os.path.basename(txt_file))[0]
        txt_file = os.path.join(folder_path, file_name + ".txt")
        report = get_teaching_report(txt_file)
        txt_file = os.path.join(out_path, file_name + ".txt")
        with open(txt_file, "w") as file:
            file.write(report)


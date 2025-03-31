import os
from openai import OpenAI
import json
import codecs

folder_path = "input"  # 文件夹路径
output_path = "output"  # 输出文件夹路径

texts = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        # 读取 .txt 文件的内容
        with open(file_path, 'r', encoding='utf-8') as file:
            contents = file.readlines()

        texts.append(contents)

text11 = """
你现在是一个教师教学语言评价专家，根据以下的四个方面(分别为教学环节、教学目标与内容、教学过程与方法、师生互动，每个方面分成四个等级，每个等级都有对应的分数)根据标准来评价教师教学语言，并且你的评价内容需要给出等级及相应等级所对应的分数(分数务必对应好是哪一个评价方面的，对应等级的对应分数也别弄错){
(1. 教学环节：等级A(分数:9~10分)：教学情境创设有新意，环节清晰，过渡自然。等级B(分数:7~8分)：教学情境创设较为新颖，环节基本清晰，过渡较为自然，但存在不足。等级C(分数:6分)：教学情境创设略显平淡，环节不够清晰，过渡有些生硬，存在一些问题。等级D(分数:0~5分)：教学情境创设平淡，环节混乱，过渡不自然，需要较大改进。)
(2. 教学目标与内容：等级A(分数:9~10分)：教学目标明确，重点突出，学习内容、任务与活动安排适宜，教学内容突出学科特色。等级B(分数:7~8分)：教学目标基本明确，重点有一定突出，学习内容、任务与活动安排基本适宜，但略有不足。等级C(分数:6分)：教学目标不够明确，重点不够突出，学习内容、任务与活动安排不够适宜，存在一些问题。等级D(分数:0~5分)：教学目标不明确，重点不突出，学习内容、任务与活动安排不适宜，需要重大改进。)
(3. 教学过程与方法：等级A(分数:9~10分)：教学方法有效针对所教内容，能照顾到学生的个别差异，注重激发学生学习兴趣与动机。等级B(分数:7~8分)：教学方法基本针对所教内容，对学生个别差异照顾较少，较重视学生学习兴趣与动机，但某些方面可能需要改进。等级C(分数:6分)：教学方法针对所教内容有些不足，对学生个别差异照顾不够，学生学习兴趣与动机不够重视，需要改进。等级D(分数:0~5分)：教学方法未针对所教内容，对学生个别差异照顾不到位，未重视学生学习兴趣与动机，存在明显不足，需要重大改进。)
(4. 师生互动：等级A(分数:9~10分)：能够利用很多提问促进学生的思考，鼓励并赞扬学生的优良表现，及时纠正学生的错误。等级B(分数:7~8分)：能够利用一定提问促进学生的思考，存在鼓励并赞扬学生的表现，但略有不足。等级C(分数:6分)：很少利用提问促进学生思考，鼓励与表扬学生不够充分，纠正学生错误不够及时，存在不足。等级D(分数:0~5分)：未能有效利用提问促进学生思考，未能鼓励与表扬学生，纠正学生错误不及时。需要重大改进。)
}请你记住这四个评价标准，并且你的评价内容根据评价标准需要给出等级及相应等级所对应的分数。

现在已经有一个评价报告模板具体内容如下（“？”表示占位符）：{
教学环节：等级：？？（对应的分数？？），评价：？？
教学目标与内容：等级：？？（对应的分数？？），评价：？？
教学过程与方法：等级：？？（对应的分数？？），评价：？？
师生互动：等级：？？（对应的分数？？），评价：？？
总体评价：？？
}这是给你的一个给你后续评价一个新的教学设计的参考模板格式，“？？”表示你要根据评价指标对教学设计进行评价的内容，每个评价的“？？”要求生成50字以上。
"""
text22 = """
现在我给你一份教师教学录音文本，请把这个教师教学录音文本根据上述的评价指标及对应的模板格式，最后来生成一个评价报告，给出每个方面（教学环节、教学目标与内容、教学过程与方法、师生互动）对应等级和相应等级的分数及评价，可以严格一点。请你根据标准适当展开评价，这对我的职业生涯很重要，下面是这个教师教学录音文本的内容：
"""
results = []
counter = 0
file_counter = 1
output_data = []

def save_to_json(data, file_counter):
    print("save_to_json------------------")
    output_file = os.path.join(output_path, f'output_{file_counter}.json')
    with codecs.open(output_file, 'a', encoding='utf-8',errors='ignore') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


client = OpenAI(api_key="your_key")

tokens = 0
print("start------------------")
for text in texts:
    completion = client.chat.completions.create( 
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": text11},
            {"role": "user", "content": text22 + str(text)}
        ]
    )
    print("completion:", completion)
    answer = completion.choices[0].message.content
    token = int(completion.usage.total_tokens)
    tokens += token
    results.append({"content": text, "evaluate": answer})
    print("counter------------------")
    print("tokens:", tokens)
    counter += 1
    if counter % 10 == 0:
        output_data.extend(results)
        save_to_json(output_data, file_counter)
        print("输出评价结果，截止第"+str(counter)+"个")
        file_counter += 1
        results = []
    # if counter == 1000:
    #     break

if results:
    output_data.extend(results)
    save_to_json(output_data, file_counter)
    print("输出评价结果，截止第" + str(counter) + "个")


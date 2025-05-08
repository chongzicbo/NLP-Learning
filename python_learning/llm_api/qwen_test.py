import os
from openai import OpenAI

chefoumula = """
\ch{Y_{2}Dy{@}I_h-{31924}c80}
\ch{Y_{2}Dy{@}D_{5h}{-}{31923}C80}
\ch{Y_{2}Dy{@}C_{2v}{-}{31922}C80}
\ch{Y_{2}Dy{@}C_1{-}{28325}C80}
\ch{Y_{2}Dy{@}c_2{-}{29591}C80}
\ch{Y_{2}Dy{@}C_{2v}{-}{31920}C80}
\ch{Y_{2}Dy{@}C_1{-}{31876}c80}
\ch{Y_{2}Dy{@}C_2{-}{28319}C80}
\ch{Y_{2}Dy{@}C_1{-}{28324}C80}
"""

message = """
我有一些chemformula 或 mhchem 包中的命令相关的latex代码，比如“\ch{Y_{2}Dy{@}I_h-{31924}C80}”，使用Latex软件渲染后，呈现的效果是：31924是下标内容，80也是下标内容。但是有些软件没法识别下标内容，因此需要使用特殊的标签将下标内容包裹起来,进行转换。
比如“\ch{Y_{2}Dy{@}I_h-{31924}C80}”转换后为“\ch{Y_{<newsup>2</newsup>}Dy{@}I_h-{<newsup>31924</newsup>}C<newsup>80</newsup>}”。
"\ch{Y_{2}Dy{@}D_{5h}{-}{31923}C80}"转换后为"\ch{Y_{<newsup>2</newsup>}Dy{@}D_{<newsup>5h</newsup>}{-}{<newsup>31923</newsup>}C<newsup>80</newsup>}" 
请注意:在latex代码中"_"下划线后接的内容是下标内容,请仔细判断哪些哪些内容是下标内容，并将其用newsup标签包裹起来。
按照上面的转换要求，请将下面的latex代码中的下标进行转换，直接返回转换后的结果，不需要解释过程：
"\ch{Y_{2}Dy{@}I_h-{31924}c80}"
"\ch{Y_{2}Dy{@}D_{5h}{-}{31923}C80}"
"\ch{Y_{2}Dy{@}C_{2v}{-}{31922}C80}"
"\ch{Y_{2}Dy{@}C_1{-}{28325}C80}"
"\ch{Y_{2}Dy{@}c_2{-}{29591}C80}"
"\ch{Y_{2}Dy{@}C_{2v}{-}{31920}C80}"
"\ch{Y_{2}Dy{@}C_1{-}{31876}c80}"
"\ch{Y_{2}Dy{@}C_2{-}{28319}C80}"
"\ch{Y_{2}Dy{@}C_1{-}{28324}C80}"
"""
try:
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-e2cc55950def4867ac0b492481b4c46d",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{message}"},
        ],
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"错误信息：{e}")
    print(
        "请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code"
    )

#####配合0312，搭一个gradio的界面去提高互动性

import pandas as pd
import json
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from modelscope import snapshot_download
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import gradio as gr
# D:/Programe/AIGC/0312/
loader = CSVLoader(file_path='test0312question.csv', encoding='gbk')
documents = loader.load()
df = pd.read_csv('test0312.csv', encoding='gbk')
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
model_dir = snapshot_download("AI-ModelScope/bge-large-zh-v1.5", revision='master')
embedding_path=model_dir
embeddings = HuggingFaceBgeEmbeddings(model_name = embedding_path)

vectorstore = FAISS.from_documents(
    docs,
    embedding= embeddings
)
retriever = vectorstore.as_retriever()

# query_vector = "历史的订单"
def qa_system(query_vector):
    top_docs = retriever.invoke(query_vector,top_k=3)

    i = 0
    formatted_data = []
    indexlist = []
    for i in range(3):
        content_lines = top_docs[i].page_content.split('\n')
        # 初始化index变量
        index = None
        # 遍历每一行，找到index所在的行
        for line in content_lines:
            if line.startswith('index:'):
                # 提取index后面的数字部分
                index = int(line.split(':')[1].strip())
                indexlist.append(index)  # 直接使用append方法
        selected_rows = df.iloc[indexlist].reset_index(drop=True)  # 使用indexlist来索引行
        
    for index, row in selected_rows.iterrows():
        # 每行数据格式化为{"question": question, "answer": answer}的形式
        formatted_data.append({"question": row["question"].replace("\n", "<br>"), "answer": row["answer"].replace("\n", "<br>")})
        # 将列表转换为JSON格式的字符串，应该在循环外部执行
          # 简化i的增加方式
    # 循环结束后，将formatted_data转换为JSON字符串
    json_result = json.dumps(formatted_data, ensure_ascii=False)
    data = json.loads(json_result)
# 创建一个新的列表来存储格式化后的结果
    formatted_result = []
    for item in data:
        formatted_item = {
            "Q": f'<span style="color: red;">{item["question"].replace("\n", "<br>")}</span>',
            "A": f'<span style="color: green;">{item["answer"].replace("\n", "<br>")}</span>'
        }
        formatted_result.append(formatted_item)
    return formatted_result

# 初始化聊天历史
chat_history = []
# 定义响应函数
def respond(message):
    bot_message = str(qa_system(message)).replace("\n", "<br>")
    chat_history.append(("User", message))
    chat_history.append(("Bot", bot_message))
    return chat_history
# 使用gr.Blocks创建界面
with gr.Blocks() as demo:
    gr.Markdown("### IPS AMP问答系统\n请输入你的问题：")
    chatbot = gr.Chatbot()  # 对话框
    msg = gr.Textbox(placeholder="请输入你的问题...")  # 输入文本框
    clear = gr.Button("清除历史")  # 清除按钮
    # 绑定输入框内的回车键的响应函数
    msg.submit(respond, inputs=msg, outputs=chatbot)
    clear.click(lambda: [], inputs=[], outputs=chatbot)
    
# def answer_question(question):
#     # 这里是你的问答系统的逻辑
#     answer = qa_system(question)
#     return answer

# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot() # 对话框
#     msg = gr.Textbox() # 输入文本框
#     clear = gr.ClearButton([msg, chatbot]) # 清除按钮
#     def respond(message, chat_history):
#         bot_message = qa_system(message)
#         chat_history.append((message, bot_message))
#         # time.sleep(2)
#         return "", chat_history
# #     # 绑定输入框内的回车键的响应函数
# #     msg.submit(respond, [msg, chatbot], [msg, chatbot])
# chat_history = []  # 初始化聊天历史
# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot()  # 对话框
#     msg = gr.Textbox()  # 输入文本框
#     clear = gr.ClearButton([msg, chatbot])  # 清除按钮
#     def respond(msg):
#         bot_message = qa_system(msg)
#         chat_history.append((msg, bot_message))
#         # time.sleep(2)  # 如果需要的话，可以取消注释这行来添加延迟
#         return chat_history
#     # 绑定输入框内的回车键的响应函数
#     msg.submit(respond, [msg, chatbot], [msg, chatbot])
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000,share=True)
    # iface.launch()
    # demo.launch(server_name="0.0.0.0", server_port=8000,share=True)
    


# iface = gr.Interface(
#     fn=qa_system, 
#     inputs="text", 
#     outputs="text",
#     title="QA 问答系统",
#     description="请输入你的问题："
# )

# if __name__ == "__main__":
#     iface.launch(server_name="0.0.0.0", server_port=8000,share=True)
    
# class Question(BaseModel):
#     query_vector: str
# app = FastAPI()
# @app.post("/")
# async def api(question: Question):
#     answer = qa_system(question.query_vector)
#     return answer
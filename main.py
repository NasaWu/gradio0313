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
        formatted_data.append({"question": row["question"], "answer": row["answer"]})
        # 将列表转换为JSON格式的字符串，应该在循环外部执行
          # 简化i的增加方式
    # 循环结束后，将formatted_data转换为JSON字符串
    json_result = json.dumps(formatted_data, ensure_ascii=False)
    return json_result


# def answer_question(question):
#     # 这里是你的问答系统的逻辑
#     answer = qa_system(question)
#     return answer

iface = gr.Interface(
    fn=qa_system, 
    inputs="text", 
    outputs="text",
    title="QA 问答系统",
    description="请输入你的问题："
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8000,share=True)
    
# class Question(BaseModel):
#     query_vector: str
# app = FastAPI()
# @app.post("/")
# async def api(question: Question):
#     answer = qa_system(question.query_vector)
#     return answer
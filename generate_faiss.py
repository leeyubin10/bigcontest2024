import os
import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# 경로 설정
data_path = './data'
module_path = './tmp'
csv_file_path = os.path.join(data_path, 'JEJU_MCT_DATA_modified_v8.csv')
index_path = os.path.join(module_path, 'faiss_index.index')
embedding_array_path = os.path.join(module_path, 'embeddings_array_file.npy')

# 모델 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "intfloat/multilingual-e5-large-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

# 텍스트 임베딩 함수 (배치 지원)
def embed_text_batch(text_batch):
    inputs = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# 전체 데이터에 대해 배치로 임베딩 생성 및 저장
def create_embeddings_and_faiss_index(batch_size=16):
    # CSV 파일에서 데이터 로드
    df = pd.read_csv(csv_file_path)
    texts = df['text'].tolist()
    
    # 임베딩과 FAISS 인덱스 준비
    dimension = 512  # 모델 임베딩 차원에 맞춰 설정
    faiss_index = faiss.IndexFlatL2(dimension)
    all_embeddings = []

    # 배치별로 텍스트 임베딩 생성
    for i in range(0, len(texts), batch_size):
        text_batch = texts[i:i+batch_size]
        embeddings = embed_text_batch(text_batch)
        all_embeddings.append(embeddings)
        faiss_index.add(embeddings)
        print(f"{i+len(text_batch)}개의 텍스트 처리 완료")

    # 모든 임베딩을 하나의 배열로 병합
    all_embeddings = np.vstack(all_embeddings)
    print(f"생성된 임베딩 배열 크기: {all_embeddings.shape}")
    print(f"FAISS 인덱스에 {faiss_index.ntotal}개의 벡터가 추가되었습니다.")

    # 파일로 저장
    faiss.write_index(faiss_index, index_path)
    np.save(embedding_array_path, all_embeddings)
    print("임베딩 배열과 FAISS 인덱스가 파일로 저장되었습니다.")

if __name__ == "__main__":
    create_embeddings_and_faiss_index(batch_size=16)  # 배치 크기 설정

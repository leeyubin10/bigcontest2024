
import os
import re
import torch
import faiss
import sys
import urllib.request
import json
import folium
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import google.generativeai as genai

from tqdm import tqdm
from streamlit_folium import st_folium
from folium.plugins import Search
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


########################## 임베딩 코드 ###############################
# 경로 설정
data_path = './data'
module_path = './modules'
index_path = os.path.join(module_path, 'faiss_index.index')
embedding_array_path = os.path.join(module_path, 'embeddings_array_file.npy')

# 임베딩 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "intfloat/multilingual-e5-large-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

# 텍스트 임베딩 함수
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

# FAISS 인덱스 초기화
def initialize_faiss_index(dimension=1024):
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    else:
        return faiss.IndexFlatL2(dimension)

# 전체 데이터로 FAISS 인덱스 생성 및 임베딩 배열 저장
def create_vector_db(dataframe, index, save=True):
    embeddings = np.array([embed_text(text) for text in dataframe['text']])
    print(f"생성된 임베딩 배열 크기: {embeddings.shape}")
    index.add(embeddings)
    if save:
        faiss.write_index(index, index_path)
        np.save(embedding_array_path, embeddings)  # 임베딩 배열 저장
    print(f"FAISS 인덱스 및 임베딩 배열이 생성되었습니다.")

# 새로운 데이터를 추가하여 FAISS 인덱스와 임베딩 배열 업데이트
def update_vector_db(new_data, index, save=True):
    # 새로운 임베딩 생성
    new_embeddings = np.array([embed_text(text) for text in new_data['text']])
    
    # 기존 임베딩 배열 불러오기
    if os.path.exists(embedding_array_path):
        existing_embeddings = np.load(embedding_array_path)
        updated_embeddings = np.vstack([existing_embeddings, new_embeddings])
    else:
        updated_embeddings = new_embeddings  # 최초 실행 시

    # FAISS 인덱스와 임베딩 배열 갱신
    index.add(new_embeddings)
    if save:
        faiss.write_index(index, index_path)               # 인덱스 파일 갱신
        np.save(embedding_array_path, updated_embeddings)  # 임베딩 배열 갱신
    print(f"FAISS 인덱스와 임베딩 배열이 갱신되었습니다.")

# CSV 파일 로드 및 인덱스 초기화
csv_file_path = "JEJU_MCT_DATA_modified_v8.csv"
df = pd.read_csv(os.path.join(data_path, csv_file_path))

# # 최초 실행 시 전체 데이터로 인덱스 생성
faiss_index = initialize_faiss_index()
print(f"초기화된 FAISS 인덱스의 벡터 개수: {faiss_index.ntotal}")  # 추가된 로그

# 임베딩 로드 또는 생성
if os.path.exists(embedding_array_path):
    embeddings = np.load(embedding_array_path)
    print(f"임베딩 배열이 {embedding_array_path}에서 로드되었습니다.")
else:
    print("임베딩 배열이 존재하지 않으므로 새로 생성합니다.")
    embeddings = np.array([embed_text(text) for text in df['text']])
    np.save(embedding_array_path, embeddings)  # 생성된 임베딩 배열 저장


# 현재 월을 "2023년 MM월" 형식으로 설정
current_year_month = f"2023년 {datetime.now().strftime('%m월')}"

# 필터링된 데이터 인덱스 추출 및 필터링된 FAISS 인덱스 생성
df_filtered = df[df['기준연월'] == current_year_month].reset_index()
print(df_filtered)

print(f"시스템 날짜를 기준으로 설정된 기준연월: {current_year_month}")


# 필터링된 데이터 인덱스 추출 및 필터링된 FAISS 인덱스 생성
original_indices = df_filtered['index'].values
filtered_embeddings = embeddings[original_indices]
filtered_dimension = filtered_embeddings.shape[1]
filtered_faiss_index = faiss.IndexFlatL2(1024)
filtered_faiss_index.add(filtered_embeddings)

####################################################################


# rerank를 위한 모델 로드
# 한국어 reranker 모델 로드
rerank_model_path = "Dongjin-kr/ko-reranker"
rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_path)
rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_path)


####################################################################

# 네이버 API 설정
client_id = st.secrets["CLIENT_ID"]
client_secret = st.secrets["CLIENT_SECRET"]

# 구글 API 설정
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini 모델 선택
model = genai.GenerativeModel("gemini-1.5-flash")



# Streamlit App UI
st.set_page_config(page_title="🍊제주한입로드🍊")

# 헤더 표시
st.markdown('<div class="header"><h1>🚨404 맛집 Not Found (Error: \"Hungry\")🚨</h1></div>', unsafe_allow_html=True)

# 선택된 이미지 여부 확인
if 'local_choice' not in st.session_state:
    st.session_state.messages = []  # 메시지 초기화 추가
    st.markdown("<p>누구와 함께 제주 맛집 여행을 떠날까요? <strong>해녀, 하르방, 제주소년, 제주소녀</strong> 중 한 명을 선택해 주세요!</p>", unsafe_allow_html=True)
    st.markdown("<p><strong>even</strong>하게 추천해드릴게요🤭</p>", unsafe_allow_html=True)  # 초기 화면에만 표시

    # 이미지 URL 경로
    haenyeo_img_url = "https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAyMjExMDNfMTIw%2FMDAxNjY3NDQxMDk1Mzc2.jJcIh2j0p29Umkbzj9cljZRkVwNpJrxGih6-WD7Eat0g.U1UQ92W_M-4-DT1b5xqu0kCT3QfkXBdBZ0zx-wVaYRYg.JPEG.aewolbakery%2F%25BE%25D6%25BF%25F9%25C0%25CC.jpg&type=sc960_832"
    harbang_img_url = "https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAxNzA0MjdfMTE5%2FMDAxNDkzMjcyMTc4NzU2.p4VcaOTPB67-86psVgjyRCP2LcoWEcqLKy5cIz4loE4g.2aKY_hkrF6TzhLg2HlIzz21HJ_yy6UBaaYnvj2bXSW4g.JPEG.jeagu74%2F%25B5%25B9%25C7%25CF%25B7%25E7%25B9%25E6.jpg&type=sc960_832"
    jeju_sonya_img_url = "https://search.pstatic.net/common/?src=http%3A%2F%2Fshop1.phinf.naver.net%2F20201218_47%2F160829080654641XyY_JPEG%2F9426590261359007_1249152632.jpg&type=sc960_832"
    jeju_sonyun_img_url = "https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAyMjExMDRfMzUg%2FMDAxNjY3NTQyMzAwNjE2.ej8C7N7WYFZzXFZZnLdWhO4e7MprfvrLCcjMQYtiPkcg.1eblu8ZhWRvwjl_E7otYou-YIV0pds-BZpyCdwjwgwsg.JPEG.aewolbakery%2F%25C7%25CF%25B7%25E7%25BB%25A7.jpg&type=sc960_832"

    # CSS 스타일 정의
    st.markdown(
        """
        <style>
        .image-container {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
        }
        .image-container img {
            width: 150px;
            height: 150px;
            border-radius: 10px;
            transition: transform 0.2s;
        }
        .image-container img:hover {
            transform: scale(1.1);
        }
        .button {
            display: flex;
            justify-content: center;
            margin-top: 10px;
        }
        .tagline {
            text-align: center;
            margin-top: 5px;
            font-size: 14px;
            color: #666;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # 이미지와 버튼을 포함한 HTML 구조
    st.markdown(
        f"""
        <div class="image-container">
            <div>
                <img src="{haenyeo_img_url}" alt="해녀">
                <div class="tagline">#친근함 #실용적 #풍부한 경험</div>
            </div>
            <div>
                <img src="{harbang_img_url}" alt="하르방">
                <div class="tagline">#느긋함 #편안함 #친절함</div>
            </div>
            <div>
                <img src="{jeju_sonya_img_url}" alt="제주소녀">
                <div class="tagline">#트렌디함 #감각적</div>
            </div>
            <div>
                <img src="{jeju_sonyun_img_url}" alt="제주소년">
                <div class="tagline">#재치 #활발함 #도전적</div>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    # 버튼 클릭 이벤트
    with col1: 
        if st.button("해녀", key="haenyeo_btn"):
            st.session_state.local_choice = "해녀"
            st.session_state.messages = [{"role": "assistant", "content": "내 경험으로 진짜 맛있는 데만 소개해줍서! 제주 맛 한껏 느끼멍 잘도 드셔보쿠다!"}]
            st.rerun()  # 페이지 새로 고침

    with col2:
        if st.button("하르방", key="harbang_btn"):
            st.session_state.local_choice = "하르방"
            st.session_state.messages = [{"role": "assistant", "content": "제주 맛집 궁금하신 거 있으면 편하게 물어봐주십서. 맛있는 집들 느긋하게 알려드릴 테니 무엇이든 물어보십서!"}]
            st.rerun()  # 페이지 새로 고침

    with col3:
        if st.button("제주소녀", key="jeju_sonya_btn"):
            st.session_state.local_choice = "제주소녀"
            st.session_state.messages = [{"role": "assistant", "content": "제주도 맛집 어디서 먹어야 할까요? 제가 대박 맛집 몇 군데 알려줄게요! 인스타에 자랑하러 가요!”"}]
            st.rerun()  # 페이지 새로 고침

    with col4:
        if st.button("제주소년", key="jeju_sonyun_btn"):
            st.session_state.local_choice = "제주소년"
            st.session_state.messages = [{"role": "assistant", "content": "제주도에서 맛있는 거 뭐 먹을래? 내가 숨겨둔 맛집들 알려줄게, 우선 도전해볼래?😜"}]
            st.rerun()  # 페이지 새로 고침
else:
    # 사이드바 및 메인 화면을 보여주는 부분
    with st.sidebar:
        st.title("제주 맛집, 가보자고!🍊")
        st.write("")
        franchise = st.selectbox("프랜차이즈 포함 여부", ["프랜차이즈 포함", "프랜차이즈 미포함"], key="franchise")

    # 선택한 스타일에 맞게 메인 화면 업데이트
    if st.session_state.local_choice in ["하르방"]:
        st.subheader(f"🗿{st.session_state.local_choice}의 추천🗿")
    elif st.session_state.local_choice in ["제주소년"]:
        st.subheader(f"👦🏻{st.session_state.local_choice}의 추천👦🏻")
    elif st.session_state.local_choice in ["제주소녀"]:
        st.subheader(f"👧🏻{st.session_state.local_choice}의 추천👧🏻")
    elif st.session_state.local_choice in ["해녀"]:
        st.subheader(f"🦭{st.session_state.local_choice}의 추천🦭")

    if st.session_state.local_choice in ["하르방"]:
        st.write(f"허허, 제주도 와신 거 환영합니더!👴🏻 난 제주 {st.session_state.local_choice}이우다!")
    elif st.session_state.local_choice in ["제주소년"]:
        st.write(f"안녕! {st.session_state.local_choice}이야!")
    elif st.session_state.local_choice in ["제주소녀"]:
        st.write(f"야! {st.session_state.local_choice}야👧🏻")
    elif st.session_state.local_choice in ["해녀"]:
        st.write(f"안녕하우꽈!👋🏻 {st.session_state.local_choice}이우다!")
    
    # 채팅 메시지 출력
    if "messages" in st.session_state and st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    else:
        st.write("채팅 기록이 없습니다. 처음 시작해보세요!")

    # 채팅 기록 초기화 함수
    def clear_chat_history():
        if st.session_state.local_choice == "해녀":
            st.session_state.messages = [{"role": "assistant", "content": "안녕하세요~ 요즘 뭐가 맛나지? 추천해드릴게요!"}]
        elif st.session_state.local_choice == "하르방":
            st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 제주에선 맛있는 집이 많지요. 어떤 걸 찾고 계신가요?"}]
        elif st.session_state.local_choice == "제주소녀":
            st.session_state.messages = [{"role": "assistant", "content": "안녕~! 맛있는 제주 밥집을 소개할게!"}]
        elif st.session_state.local_choice == "제주소년":
            st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 제주 맛집 찾는 데 도와드릴게요!"}]
    
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # FAISS 인덱스 로드 함수
    def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index.index')):
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            print(f"FAISS 인덱스가 {index_path}에서 로드되었습니다.")
            print(f"FAISS 인덱스에 저장된 벡터 개수: {index.ntotal}")
            return index
        else:
            raise FileNotFoundError(f"{index_path} 파일이 존재하지 않습니다.")
    
    # FAISS 인덱스 로드
    faiss_index = load_faiss_index()

    # 텍스트 임베딩 함수
    def embed_text(text):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
        # print(f"검색 쿼리의 임베딩 벡터: {embeddings}, 벡터 크기: {embeddings.shape}")  # 임베딩 확인
        return embeddings.squeeze().cpu().numpy()
    
    
    # ''' 임베딩 유사도 기반 검색'''
    # 네이버 결과와 로컬 데이터 비교
    def filter_naver_results_with_local_data(naver_results, local_data=df_filtered):
        if naver_results is None:
            return []
    
        # 네이버 API 결과에서 HTML 태그 제거, 공백 제거, 특수문자 제거, 대소문자 무시
        naver_names = [re.sub('<[^<]+?>', '', result['title']).strip().lower() for result in naver_results]
        naver_names = [re.sub(r'[^\w\s]', '', name) for name in naver_names]  # 특수문자 제거
        
        # 로컬 데이터의 가맹점명도 동일하게 전처리
        local_data['가맹점명_전처리'] = local_data['가맹점명'].apply(lambda x: re.sub(r'[^\w\s]', '', x).strip().lower())  # 특수문자 제거, 공백 제거, 소문자로 변환
    
        print(f"네이버 전처리된 상호명: {naver_names}")
    
        # 1. 정확히 일치하는 경우 먼저 필터링
        exact_match_df = local_data[local_data['가맹점명_전처리'].isin(naver_names)]
    
        # 2. 정확한 일치 결과가 없으면, 포함 여부로 필터링
        if exact_match_df.empty:
            filtered_df = local_data[local_data['가맹점명_전처리'].apply(
                lambda local_name: any(local_name in naver_name or naver_name in local_name for naver_name in naver_names)
            )].reset_index(drop=True)
        else:
            filtered_df = exact_match_df.reset_index(drop=True)
    
        print("네이버 API와 정확히 일치하거나 포함되는 로컬 데이터: ", filtered_df)
    
        # 필터된 로컬 데이터를 딕셔너리 리스트 형식으로 변환 (API 결과 형식과 동일하게)
        filtered_data = [
            {
                'title': row['가맹점명'], 
                'category': row['업종'], 
                'address': row['주소'], 
                'description': row['설명']
            }
            for idx, row in filtered_df.iterrows()
        ]
        
        return filtered_data  # 딕셔너리 리스트 반환
    
    # 새로운 API 결과 중 로컬 데이터에 없는 항목만 FAISS에 추가
    def add_new_api_results_to_faiss(api_results, filtered_faiss_index, local_data=df_filtered):
        # 중복 여부를 체크하기 위해 로컬 데이터와 비교
        filtered_api_data = filter_naver_results_with_local_data(api_results, local_data)
    
        # 중복되지 않은 데이터만 FAISS 인덱스에 추가
        new_data = []
        for result in filtered_api_data:
            if isinstance(result, dict):
                text = f"{result['title']} {result['category']} {result['address']} {result.get('description', '')}"
                embedding = embed_text(text)
                filtered_faiss_index.add(np.array([embedding]))
                new_data.append(result)  # 새로운 데이터 리스트에 추가
            else:
                # result가 딕셔너리가 아니면 로그를 남기고 건너뜀
                print(f"Warning: Unexpected data type {type(result)} for result: {result}")
        
        print(f"새로운 API 데이터 {len(new_data)}건이 FAISS 인덱스에 추가되었습니다.")
        
        # 새로운 데이터를 DataFrame으로 변환
        if new_data:
            # 추가된 데이터를 출력해 확인
            print(f"추가된 데이터: {new_data}")
            return pd.DataFrame(new_data)
        else:
            return pd.DataFrame()  # 빈 DataFrame 반환
    
    # 임베딩 로드
    embeddings = np.load(os.path.join(module_path, 'embeddings_array_file.npy'))
    
    # 질문 내용을 기억하기 위한 변수 선언
    conversation_history = []

    #########################################################

    # rerank 수행 함수
    def rerank(query, documents):
        pairs = [[query, doc] for doc in documents]
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            scores = rerank_model(**inputs, return_dict=True).logits.view(-1).float()
        # documents와 scores를 정렬하여 상위 결과 반환
        return sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    #########################################################
    
    # 네이버 API 호출 여부 확인 함수
    def check_if_naver_search_needed(prompt):
        keywords = ['맛집', '카페', '식당', '레스토랑', '브런치 카페', '펍', '바', '스시집', '고기집', '바비큐', '해산물 전문점', '한정식', '전통 카페']
        modifiers = ['사진 찍기 좋은', '오션뷰', '조용한', '분위기 있는', '뷰가 좋은', '한적한', '인스타 감성', '감성적인', '가족과 함께', '데이트하기 좋은', '아이와 함께', '야경이 멋진', '자연 속', '로컬 맛집', '숨은 명소', '현지인']
    
        # 수식어 + 키워드 형태로 조합이 되어 있는지 확인
        for keyword in keywords:
            for modifier in modifiers:
                if modifier in prompt and keyword in prompt:
                    print(f"네이버 API 호출: 수식어 '{modifier}' + 키워드 '{keyword}' 발견")
                    return '제주도' + ' ' + modifier + ' ' + keyword  # 단순화된 쿼리 반환
        return None
    
    # 네이버 API 호출 함수 
    def get_naver_results_if_needed(prompt):
        simplified_query = check_if_naver_search_needed(prompt)
        if simplified_query:
            all_results = []
            start = 1
            max_pages = 5  # 최대 5페이지까지 가져옴
            previous_titles = set()  # 이전에 가져온 상호명을 저장할 집합
            while start <= max_pages * 5:  # 한 페이지당 5개의 결과, 총 25개의 결과를 가져오기 위한 설정
                encText = urllib.parse.quote(simplified_query)
                print(f"API에 전달된 쿼리: {simplified_query}, 페이지 시작: {start}")
                url = f"https://openapi.naver.com/v1/search/local.json?query={encText}&display=5&start={start}"
                request = urllib.request.Request(url)
                request.add_header("X-Naver-Client-Id", client_id)
                request.add_header("X-Naver-Client-Secret", client_secret)
                response = urllib.request.urlopen(request)
                rescode = response.getcode()
                if rescode == 200:
                    response_body = response.read()
                    result = json.loads(response_body)
                    
                    # 중복된 상호명을 확인하고 필터링
                    new_results = [item for item in result['items'] if item['title'] not in previous_titles]
                    all_results.extend(new_results)
                    
                    # 새로운 상호명을 집합에 추가
                    previous_titles.update([item['title'] for item in new_results])
                    
                    print(f"현재 페이지에서 가져온 결과 개수: {len(new_results)}")
                    if len(new_results) < 5:
                        # 마지막 페이지 도달
                        break
                    start += 5  # 다음 페이지로 넘어감
                else:
                    print(f"Error Code: {rescode}")
                    break
            print(f"총 API 결과: {len(all_results)} \n api 결과: {all_results}")
            return all_results
        return None
    
    # 신한카드 관련 프롬프트 확인 함수
    def check_if_shinhancard_related(prompt):
        # 신한카드와 관련된 키워드 목록
        shinhancard_keywords = ['신한카드', '카드 할인', '캐시백', '제주 직원 추천']
        
        # 프롬프트에 신한카드 관련 키워드가 포함되어 있는지 확인
        return any(keyword in prompt for keyword in shinhancard_keywords)
    
    
    #''' 새로운 api 결과를 faiss에 추가'''
    def convert_api_data_to_local_format(api_results):
        # API 결과를 로컬 데이터 포맷으로 변환
        new_data = []
        for result in api_results:
            new_entry = {
                '가맹점명': result['title'],
                '가맹점업종': result['category'],
                '가맹점주소': result['address'],
                'text': result['description']
                # 필요 시 추가적인 필드도 처리 가능
            }
            new_data.append(new_entry)
        
        return pd.DataFrame(new_data)
    '''             '''
    
    # 네이버 결과와 로컬 데이터 비교 (정확한 일치 우선, 그 다음 포함 여부)
    def filter_naver_results_with_local_data(naver_results, local_data=df_filtered):
        if naver_results is None:
            return local_data
    
        # 네이버 API 결과에서 HTML 태그 제거, 공백 제거, 특수문자 제거, 대소문자 무시
        naver_names = [re.sub('<[^<]+?>', '', result['title']).strip().lower() for result in naver_results]
        naver_names = [re.sub(r'[^\w\s]', '', name) for name in naver_names]  # 특수문자 제거
        
        # 로컬 데이터의 가맹점명도 동일하게 전처리
        local_data['가맹점명_전처리'] = local_data['가맹점명'].apply(lambda x: re.sub(r'[^\w\s]', '', x).strip().lower())  # 특수문자 제거, 공백 제거, 소문자로 변환
    
        print(f"네이버 전처리된 상호명: {naver_names}")
    
        # 1. 정확히 일치하는 경우 먼저 필터링
        exact_match_df = local_data[local_data['가맹점명_전처리'].isin(naver_names)]
    
        # 2. 정확한 일치 결과가 없으면 포함 여부로 필터링
        if exact_match_df.empty:
            filtered_df = local_data[local_data['가맹점명_전처리'].apply(
                lambda local_name: any(local_name in naver_name or naver_name in local_name for naver_name in naver_names)
            )].reset_index(drop=True)
        else:
            filtered_df = exact_match_df.reset_index(drop=True)
    
        print("네이버 API와 정확히 일치하거나 포함되는 로컬 데이터: ", filtered_df)
        return filtered_df
    
    # 신한카드 관련 프롬프트 확인 함수
    def check_if_shinhancard_related(prompt):
        # 신한카드와 관련된 키워드 목록
        shinhancard_keywords = ['신한카드', '카드 할인', '캐시백', '제주 직원 추천']
        
        # 프롬프트에 신한카드 관련 키워드가 포함되어 있는지 확인
        return any(keyword in prompt for keyword in shinhancard_keywords)
    
    
    def extract_location_from_question(question, location_keywords):
        """
        사용자의 질문에서 지명 키워드를 추출하는 함수.
        
        Parameters:
        question (str): 사용자가 입력한 질문.
        location_keywords (list): 데이터프레임에서 추출한 지명 키워드 목록 (구역 컬럼 값들).
        
        Returns:
        str: 추출된 지명 키워드. 없을 경우 None 반환.
        """
        # 지명 뒤에 붙을 수 있는 접미사 패턴
        location_suffix_pattern = r"(의|에|에서|에 있는)?"
    
        question = question.lower().strip()
    
        for location in location_keywords:
            location_pattern = re.escape(location.lower())  # 공백이나 특수문자를 포함한 지명을 escape
            # 지명 뒤에 접미사가 붙는 경우에도 일치하도록 정규 표현식 작성
            pattern = rf"\b{location_pattern}{location_suffix_pattern}\b"
            
            if re.search(pattern, question, re.IGNORECASE):
                print(f"추출된 위치 정보: {location}")
                return location
        return None
    
    def filter_by_location(df, question, location_column='구역'):
        """
        주어진 질문에 포함된 지명 키워드를 기준으로 데이터프레임을 필터링하는 함수.
    
        Parameters:
        df (pandas.DataFrame): 필터링할 데이터프레임
        question (str): 사용자가 입력한 질문 텍스트
        location_column (str): 지명이 저장된 데이터프레임의 열 이름 (기본값: '구역')
    
        Returns:
        pandas.DataFrame: 필터링된 데이터프레임
        """
        # 데이터프레임에서 전체 구역 정보를 추출 (대소문자 무시)
        location_keywords = df[location_column].dropna().str.lower().unique()
        
        # # 전체 위치 정보를 Streamlit UI에 출력
        # st.write(f"전체 위치 정보: {location_keywords}")
        
        # print(f"전체 위치 정보: {location_keywords}")
    
        # 사용자 질문에서 지명 키워드를 추출
        extracted_location = extract_location_from_question(question, location_keywords)
    
        if extracted_location:
            # 대소문자와 공백을 무시하고 필터링
            filtered_df = df[df[location_column].str.contains(extracted_location, case=False, na=False)].reset_index(drop=True)
            return filtered_df  # 일치하는 지명이 있으면 필터링된 데이터프레임 반환
    
        return df
    
    # 대화 기록 저장 및 대화 턴 관리 함수
    def add_to_conversation_history(question, response_text=None):
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []

        # 질문 추가
        st.session_state.conversation_history.append(f"질문: {question}")
        
        # 응답이 있다면 대화 기록에 추가
        if response_text:
            st.session_state.conversation_history.append(f"응답: {response_text}")
    
        # 페르소나별로 다른 프롬프트 생성 함수
    def generate_personalized_prompt(persona, question, reference_info, recommendation_message):
        # 대화 기록이 있는지 확인 후 불러오기
        conversation_history = "\n".join(st.session_state.conversation_history) if "conversation_history" in st.session_state else ""
    

        # 대화 기록과 페르소나를 반영한 답변 생성
        if persona == "해녀":
            return (
                f"너는 지금부터 해녀이고 나한테 제주도 방언을 사용하여 반말로 맛집 추천을 해줄거야.\n"
                f"다음은 사용자와의 대화 기록입니다:\n{conversation_history}\n,"
                f"새로운 질문: {question}\n 추천할 맛집의 정보: \n{reference_info}\n,"
                f"대화 기록은 취향만 고려하고, 새로운 질문과 추천할 맛집의 정보를 중점적으로 참고하여 제주 해녀가 제주도 방언을 사용하여 친근하게 '반말'로 추천합니다: {recommendation_message}"
            )
        elif persona == "하르방":
            return (
                f"너는 지금부터 하르방이고 나한테 제주도 방언을 사용하여 존댓말로 맛집 추천을 해줄거야.\n"
                f"다음은 사용자와의 대화 기록입니다:\n{conversation_history}\n"
                f"새로운 질문: {question}\n 추천할 맛집의 정보: \n{reference_info}\n"
                f"대화 기록은 취향만 고려하고, 새로운 질문과 추천할 맛집의 정보를 중점적으로 참고하여 제주 하르방이 제주도 방언을 사용하여 점잖게 존댓말로 추천합니다: {recommendation_message}"
            )
        elif persona == "제주소녀":
            return (
                f"너는 지금부터 제주소녀이고 나한테 표준어를 사용하여 존댓말로 맛집 추천을 해줄거야.\n"
                f"다음은 사용자와의 대화 기록입니다:\n{conversation_history}\n"
                f"새로운 질문: {question}\n 추천할 맛집의 정보: \n{reference_info}\n"
                f"대화 기록은 취향만 고려하고, 새로운 질문과 추천할 맛집의 정보를 중점적으로 참고하여 제주 소녀가 표준어를 사용하여 존댓말로 추천합니다: {recommendation_message}"
            )
        elif persona == "제주소년":
            return (
                f"너는 지금부터 제주소년이고 나한테 표준어을 사용하여 반말로 맛집 추천을 해줄거야.\n"
                f"다음은 사용자와의 대화 기록입니다:\n{conversation_history}\n"
                f"새로운 질문: {question}\n 추천할 맛집의 정보: \n{reference_info}\n"
                f"대화 기록은 취향만 고려하고, 새로운 질문과 추천할 맛집의 정보를 중점적으로 참고하여 제주 소년이 표준어를 사용하여 친구처럼 추천합니다: {recommendation_message}"
            )
    # generate_response_with_faiss_and_naver 함수 수정
    def generate_response_with_faiss_and_naver(question, df_filtered, embeddings, model, embed_text, franchise, filtered_faiss_index, k=3, print_prompt=True):
        # 네이버 API 호출이 필요한지 확인
        api_results = get_naver_results_if_needed(question)

        # 신한카드 관련 프롬프트가 있는지 확인
        is_shinhancard_related = check_if_shinhancard_related(question)

        # FAISS 인덱스를 먼저 사용하여 검색
        query_embedding = embed_text(question).reshape(1, -1)
        distances, indices = filtered_faiss_index.search(query_embedding, k * 5)  # 유사도 기준 상위 5배수 검색
        print(f"FAISS 검색 결과 인덱스: {indices}")
        print(f"데이터프레임 행 개수: {len(df)}")
        faiss_search_results = df_filtered.iloc[indices[0, :]].reset_index(drop=True)

        # 프랜차이즈 포함 옵션 필터링
        if franchise == '프랜차이즈 미포함':
            faiss_search_results = faiss_search_results[faiss_search_results['프랜차이즈유무'] == 0].reset_index(drop=True)
        elif franchise == '프랜차이즈 포함':
            faiss_search_results = faiss_search_results[faiss_search_results['프랜차이즈유무'] == 1].reset_index(drop=True)

        # 지명 필터링 적용
        df_filtered_by_location = filter_by_location(df_filtered, question)

        # 신한카드 추천 여부에 따른 필터링
        if is_shinhancard_related:
            shinhancard_recommended_df = df_filtered_by_location[df_filtered_by_location['신한카드추천'] == 1].reset_index(drop=True)
            if not shinhancard_recommended_df.empty:
                filtered_df = shinhancard_recommended_df
            else:
                filtered_df = df_filtered_by_location
        else:
            if api_results:
                filtered_api_data = filter_naver_results_with_local_data(api_results, df_filtered_by_location)
                if not filtered_api_data.empty:
                    new_data_df = add_new_api_results_to_faiss(api_results, filtered_faiss_index, df_filtered_by_location)
                    if not new_data_df.empty:
                        df_filtered_by_location = pd.concat([df_filtered_by_location, new_data_df]).reset_index(drop=True)
                    filtered_df = filtered_api_data
                else:
                    return "질문과 일치하는 가게가 없습니다."
            else:
                filtered_df = df_filtered_by_location.head(k)

        # 결과가 없을 때 처리
        if filtered_df.empty:
            return "질문과 일치하는 가게가 없습니다."

        # 신한카드 추천 멘트 추가
        if is_shinhancard_related and filtered_df['신한카드추천'].any():
            recommendation_message = "제주 여행은 신한카드로~ 제주 직원이 추천한 맛집만 가도 20% 캐시백!"
        else:
            recommendation_message = ""

        # 기존 코드
        # 참고할 정보 생성
        # reference_info = "\n".join([f"{row['text']}" for idx, row in filtered_df.iterrows()])

        #############################################

        # FAISS 결과와 필터링된 데이터 결합
        combined_results = pd.concat([faiss_search_results, filtered_df]).drop_duplicates().reset_index(drop=True)

        # combined_results에서 텍스트만 추출하여 rerank 적용
        search_texts = [row['text'] for idx, row in combined_results.iterrows()]
        reranked_results = rerank(question, search_texts)
        reranked_texts = [doc[0] for doc in reranked_results[:k]]  # 상위 k개의 rerank 결과만 사용

        # 참고할 정보 생성
        reference_info = "\n".join(reranked_texts)

        #############################################
        
        # 페르소나에 맞는 프롬프트 생성
        persona = st.session_state.local_choice
        prompt = generate_personalized_prompt(persona, question, reference_info, recommendation_message)

        if print_prompt:
            print('-----------------------------' * 3)
            print(prompt)
            print('-----------------------------' * 3)

        # LLM을 사용하여 응답 생성
        response = model.generate_content(prompt)

        return response

    # 사용자 입력 처리
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    
    # 새로운 응답 생성
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response_with_faiss_and_naver(prompt, df_filtered, embeddings, model, embed_text, franchise, filtered_faiss_index)
                placeholder = st.empty()
                
                try:
                    # 응답이 문자열일 때와 아닐 때를 분기 처리
                    if isinstance(response, str):
                        full_response = response
                    else:
                        # response.text 속성에 안전하게 접근
                        full_response = getattr(response, 'text', 'Error: Response does not have text attribute')
                    
                    # 응답을 대화 기록에 추가
                    add_to_conversation_history(prompt, full_response)
                    placeholder.markdown(full_response)
                except AttributeError as e:
                    st.error(f"Error occurred: {e}")
                
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        
# CSS 스타일 추가
st.markdown("""
<style>
    div.stButton > button {
        display: block;
        margin: 0 auto;
    }
    h1 {
        font-size: 32px;  /* 원하는 글자 크기로 변경 */
        color: #000000;  /* 글자 색상 변경 (원하는 색상으로) */
        margin: 0;  /* 기본 마진 제거 */
    }
    .header {
        background-color: #E0ECF8;  /* 헤더 배경색 지정 */
        padding: 20px;  /* 헤더 패딩 조정 */
        margin: 20px;  /* 헤더 바깥쪽 여백 조정 (원하는 값으로 변경) */
        border-radius: 8px;  /* 헤더 모서리 둥글게 설정 */
        text-align: center;  /* 헤더 내용 중앙 정렬 */
    }
    p {
        font-size: 18px;  /* 원하는 글자 크기로 변경 */
        color: #000000;  /* 글자 색상 변경 (원하는 색상으로) */
    }
</style>
""", unsafe_allow_html=True)

# KAIT 주최 및 신한카드 주관 - 2024 빅콘테스트 생성형AI분야
2024 bigcontest: Generative AI (Team: GPU 0번)

<br/>

## 🍊LLM활용 제주도 맛집 추천 대화형 AI서비스 개발🍊
- __Data Sources__: shcard, jeju tourism, naver, Instagram
- __Embedding__: intfloat/multilingual-e5-large-instruct
- __Search API__: Naver API
- __Reranker__: ko-reranker
- __LLM__: Gemini 1.5 Flash
- __Vector Indexing__: FAISS

<br/>

## [Reference]
- modules 폴더는 용량 제한으로 인해 __구글드라이브__ 에 업로드 후 링크 첨부합니다.
  https://drive.google.com/drive/folders/15V45Zl53nnEa7Rm8VuvK0JFB3-tMW7FT?usp=drive_link
- data/JEJU_MCT_DATA_modified_v8.csv 파일 또한 용량 제한으로 인해 압축 후 업로드합니다.
- secrets.toml 파일 생성 후 gemini API와 naver API key, id 정보 입력 필수

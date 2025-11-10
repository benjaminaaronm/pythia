import requests
import streamlit as st


API_URL = 'http://localhost:8000/query'


def main():
    st.title('pythia - RAG Q&A demo')

    question = st.text_input('Ask a question about the loaded documents:')
    top_n = st.slider('Number of context chunks', min_value=1, max_value=10, value=5)

    if st.button('Ask') and question.strip():
        with st.spinner('Querying API...'):
            payload = {'question': question, 'top_n': top_n}
            resp = requests.post(API_URL, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            st.subheader('Answer')
            st.write(data['answer'])

            st.subheader('Retrieved context')
            for ctx in data['contexts']:
                st.markdown(f'**{ctx['title']}** (score: {ctx['score']:.2f})')
                st.write(ctx['text'])
                st.markdown('---')
        else:
            st.error(f'Error from API: {resp.text}')


if __name__ == '__main__':
    main()

import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="Translator (EN ↔ BM)",
    layout="centered",
    initial_sidebar_state="collapsed"
)

MODEL_NAME = "mesolitica/t5-base-standard-bahasa-cased"

@st.cache_resource
def load_client():
    token = st.secrets.get("HF_TOKEN", None)
    return InferenceClient(token=token)

client = load_client()

LANG_EN = "English"
LANG_BM = "Bahasa Malaysia"
available_languages = [LANG_EN, LANG_BM]

if 'translate_history' not in st.session_state:
    st.session_state['translate_history'] = []
if 'translated_text' not in st.session_state:
    st.session_state['translated_text'] = ""
if 'source_lang' not in st.session_state:
    st.session_state['source_lang'] = LANG_EN

st.title("English ↔ Bahasa Malaysia Translator")

col1_lang, col2_swap, col3_lang = st.columns([1, 0.2, 1])

with col1_lang:
    def on_source_lang_change():
        st.session_state['source_lang'] = st.session_state['source_lang_select']
        st.session_state['translated_text'] = ""

    source_lang = st.selectbox(
        "Source Language",
        available_languages,
        index=available_languages.index(st.session_state['source_lang']),
        key="source_lang_select",
        on_change=on_source_lang_change
    )

with col2_swap:
    st.write("")
    st.write("")
    if st.button("↔", help="Swap languages", key="swap_button"):
        current = st.session_state['source_lang']
        new_lang = LANG_BM if current == LANG_EN else LANG_EN
        st.session_state['source_lang'] = new_lang
        st.session_state['source_lang_select'] = new_lang
        st.session_state['translated_text'] = ""
        st.rerun()

with col3_lang:
    source_lang = st.session_state['source_lang']
    target_lang = LANG_BM if source_lang == LANG_EN else LANG_EN
    st.text_input(
        label="Target Language",
        value=target_lang,
        disabled=True,
        key="target_lang_display"
    )

st.markdown("---")

input_col, output_col = st.columns(2)

with input_col:
    input_placeholder = f"Enter text in {source_lang}:"
    default_input_text = "I love you." if source_lang == LANG_EN else "Saya sayang awak."

    input_text = st.text_area(
        input_placeholder,
        default_input_text,
        height=200,
        key="input_text_area"
    )

with output_col:
    st.text_area(
        f"{target_lang} Translation:",
        value=st.session_state['translated_text'],
        height=200,
        disabled=True
    )

if source_lang == LANG_EN and target_lang == LANG_BM:
    t5_prefix = "terjemah Inggeris ke Melayu:"
elif source_lang == LANG_BM and target_lang == LANG_EN:
    t5_prefix = "terjemah Melayu ke Inggeris:"
else:
    st.error("Invalid language combination for translation.")
    st.stop()

if st.button("Translate", key="translate_button", use_container_width=True):
    if not input_text:
        st.warning(f"Please enter some text in {source_lang} to translate.")
    else:
        try:
            with st.spinner("Translating..."):
                processed_input = f"{t5_prefix} {input_text}"
                result = client.text2text_generation(
                    processed_input,
                    model=MODEL_NAME,
                    max_new_tokens=150,
                )
                translated_text = result if isinstance(result, str) else result[0].generated_text

            st.session_state['translate_history'].append({
                "source_lang": source_lang,
                "target_lang": target_lang,
                "source_text": input_text,
                "translated_text": translated_text
            })
            st.session_state['translate_history'] = st.session_state['translate_history'][-10:]
            st.session_state['translated_text'] = translated_text
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred during translation: {e}")

with st.expander("View Translation History"):
    if st.session_state['translate_history']:
        for i, entry in enumerate(st.session_state['translate_history']):
            st.markdown(
                f"{i+1}. From {entry['source_lang']} to {entry['target_lang']}:<br>"
                f"**Source:** {entry['source_text']}<br>"
                f"**Translation:** {entry['translated_text']}",
                unsafe_allow_html=True
            )
        if st.button("Clear History", key="clear_history_button"):
            st.session_state['translate_history'] = []
            st.session_state['translated_text'] = ""
            st.rerun()
    else:
        st.info("No translation history yet.")

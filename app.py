import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

torch.classes.__path__ = [] 
st.set_page_config(
    page_title="Translator (EN ↔ BM)",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Model Loading 
@st.cache_resource
def load_model():
    model_name = "mesolitica/t5-base-standard-bahasa-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Languages Available 
LANG_EN = "English"
LANG_BM = "Bahasa Malaysia"
available_languages = [LANG_EN, LANG_BM]

# Initialize Session State
if 'translate_history' not in st.session_state:
    st.session_state['translate_history'] = []
if 'translated_text' not in st.session_state:
    st.session_state['translated_text'] = ""
# FIX: Track source lang directly in session state (not just index)
if 'source_lang' not in st.session_state:
    st.session_state['source_lang'] = LANG_EN

# Streamlit Title 
st.title("English ↔ Bahasa Malaysia Translator")

# Language Selection and Swap 
col1_lang, col2_swap, col3_lang = st.columns([1, 0.2, 1])

with col1_lang:
    # FIX: Use on_change to sync widget value back to session state,
    # and set value= from session state so swap can control it.
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
        # FIX: Update BOTH the backing state AND the widget key so Streamlit
        # re-renders the selectbox at the correct index.
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

# Create two columns for the input and output text areas
input_col, output_col = st.columns(2)

with input_col:
    input_placeholder = f"Enter text in {source_lang}:"
    if source_lang == LANG_EN:
        default_input_text = "I love you."
    else: 
        default_input_text = "Saya sayang awak."

    input_text = st.text_area(
        input_placeholder,
        default_input_text,
        height=200, 
        key="input_text_area"
    )

with output_col:
    # FIX: Don't use `key` on the output text_area — use `value` directly from
    # session state. A keyed disabled widget won't reflect session state updates
    # after its first render.
    st.text_area(
        f"{target_lang} Translation:",
        value=st.session_state['translated_text'],
        height=200,
        disabled=True
    )

# Dynamic T5 Prefix 
if source_lang == LANG_EN and target_lang == LANG_BM:
    t5_prefix = "terjemah Inggeris ke Melayu:"
elif source_lang == LANG_BM and target_lang == LANG_EN:
    t5_prefix = "terjemah Melayu ke Inggeris:"
else:
    st.error("Invalid language combination for translation.")
    st.stop()

# Translate Button 
if st.button("Translate", key="translate_button", use_container_width=True):
    if not input_text:
        st.warning(f"Please enter some text in {source_lang} to translate.")
    else:
        try:
            with st.spinner("Translating..."):
                processed_input = f"{t5_prefix} {input_text}"
                inputs = tokenizer(processed_input, return_tensors="pt", max_length=512, truncation=True)
                
                if torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}

                generation_kwargs = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "max_length": 150,
                    "num_beams": 5,
                    "early_stopping": True
                }

                outputs = model.generate(**generation_kwargs)
                translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.session_state['translate_history'].append({
                "source_lang": source_lang,
                "target_lang": target_lang,
                "source_text": input_text,
                "translated_text": translated_text
            })
            st.session_state['translate_history'] = st.session_state['translate_history'][-10:]

            # FIX: Write result to session state, then rerun — the output
            # text_area above reads directly from session state without a key,
            # so it will correctly reflect the new value after rerun.
            st.session_state['translated_text'] = translated_text
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred during translation: {e}")

# Translation History
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
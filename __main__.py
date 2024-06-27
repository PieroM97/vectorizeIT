import streamlit as st
import pandas as pd

from common.loader import load_file
from core.embed import vectorize_text
from core.config import MODEL_OPTIONS

from transformers import AutoTokenizer, AutoModel

st.title("VectorizeIT")

uploaded_file = st.file_uploader("Drag and drop a file here")


if uploaded_file is not None:
    df = load_file(uploaded_file)
    original_filename = uploaded_file.name

    if df is not None:
        # Create a three-column layout
        col1, col2, col3 = st.columns(3)

        # First column: Selectbox for column selection
        with col1:
            selected_column = st.selectbox("Select a column to vectorize", df.columns)

        # Second column: Slider for row selection
        with col2:
            # Model selection
            selected_model = st.selectbox("Select a model", MODEL_OPTIONS)


        # Third column: Checkbox for quantization
        with col3:
            num_rows = len(df)
            rows_to_process = st.slider("Select number of rows to process", 1, num_rows, num_rows)
            quantize = st.checkbox("Quantize Embeddings")

        # Button to start vectorization
        if st.button("Start"):
            progress_bar = st.progress(0)
            progress_text = st.empty()  # Placeholder for progress text

            # Load the selected model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(selected_model)
            model = AutoModel.from_pretrained(selected_model)

            vectors = []
            for i, text in enumerate(df[selected_column][:rows_to_process]):
                vector = vectorize_text(str(text), tokenizer, model, quantize=quantize)
                vectors.append(vector)

                # Update progress bar and text
                progress_percent = (i + 1) / rows_to_process
                progress_bar.progress(progress_percent)
                progress_text.text(f"Vectorization progress: {int(progress_percent * 100)}%")

            progress_text.text(f"Vectorization complete")

            # Add vectors to DataFrame
            df.loc[:rows_to_process-1, 'embedding'] = pd.Series(vectors)

            df.to_json(f"output/{original_filename}_vectorized.json", orient='records')

    else:
        st.error("Failed to load file.")

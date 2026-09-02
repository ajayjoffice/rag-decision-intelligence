import sys
import streamlit as st
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from rag.pipeline import RAGPipeline


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RAG Decision Intelligence",
    page_icon="🔎",
    layout="wide"
)


# ==========================================
# LOAD RAG PIPELINE
# ==========================================

@st.cache_resource
def load_pipeline():
    return RAGPipeline()


pipeline = load_pipeline()


# ==========================================
# HEADER
# ==========================================

st.title("RAG Decision Intelligence")

st.markdown(
    """
    **Enterprise Policy Question Answering**

    Ask questions about company policies, benefits,
    work practices, career development, leave, devices,
    and other internal documentation.
    """
)

st.divider()


# ==========================================
# QUESTION INPUT
# ==========================================

question = st.text_area(
    "Ask a question",
    placeholder="Example: Can employees work on outside projects?",
    height=100
)

ask_button = st.button(
    "Ask Question",
    type="primary",
    use_container_width=True
)


# ==========================================
# ANSWER
# ==========================================

if ask_button:

    if not question.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner("Retrieving evidence and generating answer..."):

            result = pipeline.answer(
                question.strip(),
                top_k=5
            )

        st.subheader("Answer")

        st.write(result["answer"])

        st.divider()


        # ==========================================
        # SYSTEM INFORMATION
        # ==========================================

        st.subheader("Retrieval Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Evidence Quality",
                result["evidence"]["quality"].title()
            )

        with col2:
            correction = (
                "Yes"
                if result["correction_used"]
                else "No"
            )

            st.metric(
                "CRAG Correction",
                correction
            )

        with col3:
            st.metric(
                "Sources",
                len(result["sources"])
            )


        # ==========================================
        # REWRITTEN QUERY
        # ==========================================

        if result["rewritten_query"]:

            st.markdown("**Corrected Search Query**")

            st.code(
                result["rewritten_query"],
                language="text"
            )


        # ==========================================
        # SOURCES
        # ==========================================

        st.subheader("Retrieved Sources")

        for i, source in enumerate(
            result["sources"],
            start=1
        ):

            document = source.get(
                "document_name",
                "Unknown document"
            )

            section = source.get(
                "section",
                "Unknown section"
            )

            with st.expander(
                f"Source {i}: {document} — {section}"
            ):

                st.markdown(
                    f"**Document:** `{document}`"
                )

                st.markdown(
                    f"**Section:** `{section}`"
                )

                if "final_score" in source:
                    st.markdown(
                        f"**RRF Score:** "
                        f"{source['final_score']:.6f}"
                    )

                if "reranker_score" in source:
                    st.markdown(
                        f"**Reranker Score:** "
                        f"{source['reranker_score']:.6f}"
                    )

                st.markdown("**Evidence:**")

                st.write(
                    source.get(
                        "text",
                        "No source text available."
                    )
                )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "RAG Decision Intelligence • "
    "Dense Retrieval + BM25 + RRF + BGE Reranking + CRAG + Local LLM"
)

from llama_cpp import Llama
from pathlib import Path


MODEL_PATH = Path(
    "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)


class LocalLLM:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        print("Loading local LLM...")

        self.llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=-1,
            verbose=False
        )

        print("Local LLM loaded successfully.")

    def generate(
        self,
        question: str,
        retrieved_chunks: list
    ):

        context_parts = []

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            context_parts.append(
                f"""
SOURCE {i}
Document: {chunk['document_name']}
Section: {chunk['section']}

{chunk['text']}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""<|start_header_id|>system<|end_header_id|>

You are an enterprise policy assistant.

Answer the user's question ONLY using the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume facts.
3. If the context does not contain enough information, clearly say that the information was not found in the provided documents.
4. Keep the answer concise and directly relevant.
5. Mention the supporting document and section when appropriate.

<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT:
{context}

QUESTION:
{question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

        output = self.llm(
            prompt,
            max_tokens=400,
            temperature=0.1,
            top_p=0.9,
            stop=[
                "<|eot_id|>",
                "<|end_of_text|>"
            ]
        )

        return output["choices"][0]["text"].strip()

    def rewrite_query(
        self,
        question: str
    ):

        prompt = f"""<|start_header_id|>system<|end_header_id|>

You are an enterprise document retrieval query optimizer.

Rewrite the user's question into ONE concise search query designed
to retrieve the exact passage containing the answer.

Rules:
1. Preserve the user's original intent and important intent words such as allowed, prohibited, required, eligibility, or consequences.
2. Identify the key concept being asked about.
3. Identify likely policy or document terminology that may be used in the source documents, including domain-specific synonyms for the user's wording.
4. Prefer specific policy terms and domain-specific synonyms over generic wording.
5. Do not remove important words describing the user's intent.
6. Do not answer the question.
7. Do not invent facts.
8. Do not use quotation marks.
9. Do not use OR, AND, or multiple alternative queries.
10. Return ONLY one search query.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Question:
{question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

        output = self.llm(
            prompt,
            max_tokens=80,
            temperature=0.0,
            stop=[
                "<|eot_id|>",
                "<|end_of_text|>"
            ]
        )

        rewritten = output["choices"][0]["text"].strip()

        rewritten = (
            rewritten
            .replace('"', "")
            .replace("'", "")
            .replace("\n", " ")
            .strip()
        )

        return rewritten
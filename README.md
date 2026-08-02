# AS Intelligence Studio

AS is a monochrome evidence-grounded Streamlit product for analyzing a fixed four-book system library, user-uploaded PDF/Word/Excel/image files, or both in a Combined Workspace.

## Final Product

- System Library, My Uploaded Files, and Combined Workspace
- Cross-source comparison with separate `[System N]` and `[Upload N]` citations
- Multimodal text, table, and visual analysis
- Real locked exams with hidden answers and rubrics
- Exact MCQ and True/False grading
- Evidence-bounded written-answer grading with partial credit
- Full results, weak-topic retakes, and JSON export
- Pure black/white/gray interface with animated evidence graphics
- Dark and light monochrome appearances

## Run

1. Place the four exact approved PDFs inside `data/documents/`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Configure Streamlit Secrets.
4. Run `streamlit run as_streamlit_app.py`.

`streamlit_app.py` is synchronized to the same final product for compatibility with older deployments.

## Streamlit Community Cloud

- Main file: `as_streamlit_app.py` (recommended) or `streamlit_app.py`
- Python: `3.12`
- Secrets:

```toml
OPENROUTER_API_KEY = "your_openrouter_key_here"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

Never commit a real API key, `.env`, or `.streamlit/secrets.toml`.
Do not publish book PDFs unless you have the legal right to redistribute them.

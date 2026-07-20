import re


def load_combined_papers(txt_path: str):
    """
    Read the literature summary file combined_texts.txt
    and parse it into a structured list of dictionaries containing metadata (literature source) and content (body text).
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        return "File not found. Please check the path.", []

    pattern = r"={60}\n## Paper Source:\s*(.*?)\n={60}\n"

    parts = re.split(pattern, raw_text)

    global_meta = parts[0].strip()

    papers_data = []

    for i in range(1, len(parts), 2):
        source_name = parts[i].strip()
        paper_content = parts[i + 1].strip()

        papers_data.append({
            "metadata": {"source": source_name},
            "text": paper_content
        })

    return global_meta, papers_data
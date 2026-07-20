# AI for Inclusive Education Design Guidance System 

## Setup & Usage

### 1. Configure your API key and model

Open `main.py` and locate the LLM configuration block (around line 14–24):

```python
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",  # change if using a different provider
    model="openai/gpt-5-mini",                # change to your desired model
    temperature=0
)
```

- **API Key**: Replace the placeholder with your key .
- **Base URL & Model**: Adjust `base_url` and `model` to match your provider (OpenAI, Azure, local LLM, etc.).

### 2. Install dependencies

```bash
pip install langchain langchain-openai langgraph scikit-learn numpy pydantic
```

### 3. Run the pipeline

```bash
python main.py
```

Enter your query when prompted — for example:

```
I want to design a MOOC plugin for people with ADHD to help lower their cognitive load. Do you have any suggestions?
```

The system will output a structured system design guideline citing relevant literature.

## Project Architecture

### `main.py`
Entry point and core pipeline. Defines the `GraphState`, a 3-node LangGraph workflow.

- **Agent 1 — Classifier**: Classifies the user query into one of 18 inclusion taxonomy categories using a structured-output LLM call.
- **Agent 2 — Retriever & Summarizer**: Loads the relevant pre-processed literature corpus, performs TF-IDF retrieval, and summarizes the top-5 papers via LLM.
- **Agent 3 — Architect**: Generates final system design guidelines using a category-specific prompt template that references the retrieved literature.

### `Agent3Prompt.py`
Contains 18 category-specific system prompts, each with tailored design considerations for students, teachers, and administrators.

### `LoadPapers.py`
Utility for parsing `combined_texts.txt` files into structured paper dictionaries (source metadata + body text).

### `Included Papers/`
A complete directory structured by the SEND taxonomy, comprising all original literature PDF files along with cleaned, 
aggregated text files (`combined_texts.txt`) for each subcategory:

```
Included Papers/
├── Disability-Based Inclusion/
│   ├── General/
│   ├── Neuro-developmental Disorder/  (ASD, ADHD, ID, SLD)
│   ├── Sensory Impairment/            (Visual, Hearing)
│   └── Motor Impairment/
├── Sociocultural-Based Inclusion/     (Gender, Language, Race & Ethnicity)
├── Socioeconomic-Based Inclusion/
├── Geographic-Based Inclusion/
├── Giftedness-Based Inclusion/
└── Lifestage-Based Inclusion/
```

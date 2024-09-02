

# Table

| Metric | Description | Input | Output | Default Settings |
| --- | --- | --- | --- | --- |
| `hf_pipeline` | Use a Hugging Face pipeline with a given task and model to evaluate a text sample. Returns the first response from the pipeline. | `text: str, task: str = None, model: str = None` | `response: Any` | None |
| `hf_label_score_aggregator` | Use a Hugging Face pipeline with a given task and model to evaluate a text sample. Returns the first response from the pipeline. | `values: list[dict[str, Any]]` | `response: dict[str, Any]` | None |
| `allm_rating_json` | Use an LLM to get a rating for a text sample. | `messages: list[OpenAIMessage]`, `criteria: str = None` | `rating: float` | None |
| `llm_pairwise_judge` | Use an LLM to judge a pairwise comparison between two text samples. | `choices: list[str]`, `criteria: str = None` | `best_choice: str`, `worst_choice: str` | None |
| `llm_choice_judge` | Use an LLM to judge a choice from a list of text samples. | `choices: list[str]`, `criteria: str = None` | `best_choice: str` | None |
| `allm_rating_json` | Use an LLM to get a rating for a text sample. | `messages: list[OpenAIMessage]`, `criteria: str = None` | `rating: float` | None |
| `llm_pairwise_judge` | Use an LLM to judge a pairwise comparison between two text samples. | `choices: list[str]`, `criteria: str = None` | `best_choice: str`, `worst_choice: str` | None |
| `llm_choice_judge` | Use an LLM to judge a choice from a list of text samples. | `choices: list[str]`, `criteria: str = None` | `best_choice: str` | None |
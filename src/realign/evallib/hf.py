from realign.evaluators import evaluator
from realign.utils import try_import


transformers = try_import('transformers')
pipeline = try_import(from_module='transformers', import_names='pipeline')
torch = try_import('torch')

@evaluator
def hf_pipeline(text, task=None, model=None):
    """
    Use a Hugging Face pipeline with a given task and model to evaluate a text sample.

    Args:
        text (str): The input text to be processed by the pipeline.
        task (str, optional): The task to be performed by the pipeline (e.g., 'sentiment-analysis', 'text-classification').
        model (str, optional): The name or path of the model to be used in the pipeline.

    Returns:
        Any: The first response from the pipeline, typically a dictionary containing the prediction results.

    Raises:
        AssertionError: If either task or model is not specified.

    Note:
        This function attempts to use MPS (Metal Performance Shaders) for GPU acceleration on Mac Apple Silicon devices.
        If MPS is not available, it falls back to using the CPU.
    """
    assert task and model, 'task and model must be specified'

    # Check if MPS is available (Mac Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) device for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    # Create the pipeline with the specified device
    pipe = pipeline(task=task, 
                    model=model,
                    device=device)

    # get the response
    print(f'\nRunning hf_pipeline with task {task} and model {model}\n')
    response = pipe(text)

    return response[0]

@evaluator
def hf_label_score_aggregator(values: dict):
    """
    Aggregate labels and scores from a list of Hugging Face pipeline outputs.

    This function takes a list of dictionaries containing 'label' and 'score' keys,
    typically output from a Hugging Face classification pipeline. It aggregates
    these results to determine the most common label, the confidence in that label,
    and the average score for that label.

    Args:
        values (list[dict]): A list of dictionaries, each containing 'label' and 'score' keys.

    Returns:
        dict: A dictionary containing:
            - 'best_label' (str): The most common label among the inputs.
            - 'label_confidence' (float): The proportion of inputs that have the best label.
            - 'average_score' (float): The average score for the best label.

    Example:
        >>> results = [
        ...     {'label': 'positive', 'score': 0.9},
        ...     {'label': 'positive', 'score': 0.8},
        ...     {'label': 'negative', 'score': 0.6}
        ... ]
        >>> hf_label_score_aggregator(results)
        {
            'best_label': 'positive',
            'label_confidence': 0.6666666666666666,
            'average_score': 0.85
        }
    """
    
    from collections import Counter
    
    # Count the occurrences of each label
    label_counts = Counter(value['label'] for value in values)
    
    # Find the most common label
    best_label, count = label_counts.most_common(1)[0]
    
    # Calculate the confidence
    label_confidence = count / len(values)
    
    # Find the average score for the best label
    best_label_scores = [value['score'] for value in values if value['label'] == best_label]
    average_score = sum(best_label_scores) / len(best_label_scores)
    
    return {
        'best_label': best_label,
        'label_confidence': label_confidence,
        'average_score': average_score,
    }

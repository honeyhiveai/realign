from transformers import pipeline
import torch

from realign.evaluators import evaluator

@evaluator
def hf_pipeline(tweet_text, task=None, model=None):
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
    response = pipe(tweet_text)

    return response[0]

@evaluator
def hf_label_score_aggregator(values):
    
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

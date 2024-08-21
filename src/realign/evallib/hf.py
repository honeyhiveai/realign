from transformers import pipeline
import torch

from realign.evaluators import evaluator

def hf_pipeline(text, task=None, model=None):
    if not task: return None
    
    # Check if MPS is available (Mac Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) device for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    # Create the pipeline with the specified device
    pipe = pipeline(task, model=model, device=device)
    
    response = pipe(text)
    
    return response[0]

@evaluator
def hf_hate_speech(text, task=None, model=None):
    return hf_pipeline(text, task, model)

@evaluator
def hf_sentiment_classifier(text, task=None, model=None):
    return hf_pipeline(text, task, model)

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

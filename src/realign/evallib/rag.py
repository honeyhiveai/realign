from datasets import Dataset 

from realign.evaluators import evaluator
from realign.utils import try_import


@evaluator
def ragas_pipeline(values: tuple[dict[str, str]], metrics: list[str] | str | None = None):
    
    if metrics is None:
        return None
    
    evaluate = try_import(from_module='ragas', import_names='evaluate')
    
    if isinstance(metrics, str):
        metrics = [metrics]
    
    metric_modules = []
    for metric in metrics:
        metric_module = try_import(from_module=f'ragas.metrics', import_names=metric)
        if metric_module is not None:
            metric_modules.append(metric_module)
        else:
            print(f"Warning: Metric '{metric}' not found in ragas.metrics. Skipping.")
    
    if not metric_modules:
        return None
    
    dataset = Dataset.from_dict(values)
    score = evaluate(dataset, metrics=metric_modules)
    score.to_pandas()
    return score



if __name__ == '__main__':

    data_samples = {
        'contexts': [["A company is launching a new product, a smartphone app designed to help users track their fitness goals. The app allows users to set daily exercise targets, log their meals, and track their water intake. It also provides personalized workout recommendations and sends motivational reminders throughout the day."]],
        'summary': ['A company is launching a fitness tracking app that helps users set exercise goals, log meals, and track water intake, with personalized workout suggestions and motivational reminders.'],
        'answer': ['A fitness tracking app that helps users set exercise goals, log meals, and track water intake, with personalized workout suggestions and motivational reminders.'],
    }

    # print(ragas_pipeline(data_samples, metrics='summarization_score'))
    print(evaluator['ragas_context_recall'](data_samples))
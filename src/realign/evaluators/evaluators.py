import re
import json
import xmltodict
import sqlparse
import Levenshtein
from rouge import Rouge
import requests
from typing import List, Callable, Dict, Any
from realign.evaluators.schema_evaluators import Eval, EvalResult

class Contains(Eval):
    '''An evaluator that checks if the input value is in the target'''

    def __init__(self, target: str):
        super().__init__(eval_type='contains', target=target, checker='contains_all')

    def __call__(self, input_value: str) -> EvalResult:
        return super().__call__(input_value)

class ContainsAll(Eval):
    def __init__(self, target: List[str]):
        super().__init__(eval_type='contains_all', target=target, checker='contains_all')

    def __call__(self, input_value: str) -> EvalResult:
        return super().__call__(input_value)

class ContainsAny(Eval):
    def __init__(self, target: List[str]):
        super().__init__(eval_type='contains_any', target=target, checker='contains_any')

    def __call__(self, input_value: str) -> EvalResult:
        return super().__call__(input_value)

class ContainsJson(Eval):
    def __init__(self, schema: Dict = None):
        super().__init__(eval_type='contains_json', target=schema)

    def __call__(self, input_value: str) -> EvalResult:
        try:
            json_data = json.loads(input_value)
            if self.target:
                from jsonschema import validate
                validate(instance=json_data, schema=self.target)
            return EvalResult(True, json_data)
        except json.JSONDecodeError:
            return EvalResult(False, None)
        except Exception:
            return EvalResult(False, None)

class ContainsSql(Eval):
    def __call__(self, input_value: str) -> EvalResult:
        try:
            parsed = sqlparse.parse(input_value)
            return EvalResult(True, parsed)
        except Exception:
            return EvalResult(False, None)

class ContainsXml(Eval):
    def __call__(self, input_value: str) -> EvalResult:
        try:
            xml_dict = xmltodict.parse(input_value)
            return EvalResult(True, xml_dict)
        except Exception:
            return EvalResult(False, None)

class Cost(Eval):
    def __init__(self, threshold: float):
        super().__init__(eval_type='cost', target=threshold)

    def __call__(self, input_value: float) -> EvalResult:
        return EvalResult(input_value <= self.target, input_value)

class Equals(Eval):
    def __init__(self, target: Any):
        super().__init__(eval_type='equals', target=target, checker='exact')

    def __call__(self, input_value: Any) -> EvalResult:
        return super().__call__(input_value)

class IContains(Eval):
    def __init__(self, target: str):
        super().__init__(eval_type='icontains', target=target.lower())

    def __call__(self, input_value: str) -> EvalResult:
        return EvalResult(self.target in input_value.lower(), input_value)

class IContainsAll(Eval):
    def __init__(self, target: List[str]):
        super().__init__(eval_type='icontains_all', target=[t.lower() for t in target])

    def __call__(self, input_value: str) -> EvalResult:
        lower_input = input_value.lower()
        return EvalResult(all(t in lower_input for t in self.target), input_value)

class IContainsAny(Eval):
    def __init__(self, target: List[str]):
        super().__init__(eval_type='icontains_any', target=[t.lower() for t in target])

    def __call__(self, input_value: str) -> EvalResult:
        lower_input = input_value.lower()
        return EvalResult(any(t in lower_input for t in self.target), input_value)

class IsJson(Eval):
    def __init__(self, schema: Dict = None):
        super().__init__(eval_type='is_json', target=schema)

    def __call__(self, input_value: str) -> EvalResult:
        try:
            json_data = json.loads(input_value)
            if self.target:
                from jsonschema import validate
                validate(instance=json_data, schema=self.target)
            return EvalResult(True, json_data)
        except Exception:
            return EvalResult(False, None)

class IsSql(Eval):
    def __init__(self, allowed_statements: List[str] = None):
        super().__init__(eval_type='is_sql', target=allowed_statements)

    def __call__(self, input_value: str) -> EvalResult:
        try:
            parsed = sqlparse.parse(input_value)
            if self.target:
                stmt_type = parsed[0].get_type().lower()
                if stmt_type not in [s.lower() for s in self.target]:
                    return EvalResult(False, None)
            return EvalResult(True, parsed)
        except Exception:
            return EvalResult(False, None)

class IsValidOpenAIFunctionCall(Eval):
    def __init__(self, function_schema: Dict):
        super().__init__(eval_type='is_valid_openai_function_call', target=function_schema)

    def __call__(self, input_value: Dict) -> EvalResult:
        try:
            from jsonschema import validate
            validate(instance=input_value, schema=self.target)
            return EvalResult(True, input_value)
        except Exception:
            return EvalResult(False, None)

class IsValidOpenAIToolsCall(Eval):
    def __init__(self, tools_schema: List[Dict]):
        super().__init__(eval_type='is_valid_openai_tools_call', target=tools_schema)

    def __call__(self, input_value: List[Dict]) -> EvalResult:
        try:
            from jsonschema import validate
            for tool_call in input_value:
                tool_schema = next((tool for tool in self.target if tool['name'] == tool_call['name']), None)
                if not tool_schema:
                    return EvalResult(False, None)
                validate(instance=tool_call, schema=tool_schema)
            return EvalResult(True, input_value)
        except Exception:
            return EvalResult(False, None)

class IsXml(Eval):
    def __call__(self, input_value: str) -> EvalResult:
        try:
            xml_dict = xmltodict.parse(input_value)
            return EvalResult(True, xml_dict)
        except Exception:
            return EvalResult(False, None)

class Javascript(Eval):
    def __init__(self, js_function: str):
        super().__init__(eval_type='javascript', target=js_function)

    def __call__(self, input_value: str) -> EvalResult:
        # Note: This would require a JavaScript runtime to be implemented properly
        raise NotImplementedError("JavaScript evaluation is not implemented in this Python-only environment")

class Latency(Eval):
    def __init__(self, threshold: float):
        super().__init__(eval_type='latency', target=threshold)

    def __call__(self, input_value: float) -> EvalResult:
        return EvalResult(input_value <= self.target, input_value)

class Levenshtein(Eval):
    def __init__(self, target: str, threshold: int):
        super().__init__(eval_type='levenshtein', target=(target, threshold))

    def __call__(self, input_value: str) -> EvalResult:
        distance = Levenshtein.distance(input_value, self.target[0])
        return EvalResult(distance <= self.target[1], distance)

class PerplexityScore(Eval):
    def __init__(self):
        super().__init__(eval_type='perplexity_score')

    def __call__(self, input_value: float) -> EvalResult:
        # Note: This is a placeholder. Actual implementation would depend on how you calculate normalized perplexity
        return EvalResult(True, input_value)

class Perplexity(Eval):
    def __init__(self, threshold: float):
        super().__init__(eval_type='perplexity', target=threshold)

    def __call__(self, input_value: float) -> EvalResult:
        return EvalResult(input_value <= self.target, input_value)

class Python(Eval):
    def __init__(self, python_function: Callable):
        super().__init__(eval_type='python', target=python_function)

    def __call__(self, input_value: Any) -> EvalResult:
        try:
            result = self.target(input_value)
            return EvalResult(bool(result), result)
        except Exception as e:
            return EvalResult(False, str(e))

class Regex(Eval):
    def __init__(self, pattern: str):
        super().__init__(eval_type='regex', target=re.compile(pattern))

    def __call__(self, input_value: str) -> EvalResult:
        match = self.target.search(input_value)
        return EvalResult(bool(match), match.group() if match else None)

class RougeN(Eval):
    def __init__(self, target: str, n: int, threshold: float):
        super().__init__(eval_type='rouge_n', target=(target, n, threshold))
        self.rouge = Rouge()

    def __call__(self, input_value: str) -> EvalResult:
        scores = self.rouge.get_scores(input_value, self.target[0])
        rouge_n_score = scores[0][f'rouge-{self.target[1]}']['f']
        return EvalResult(rouge_n_score >= self.target[2], rouge_n_score)

class StartsWith(Eval):
    def __init__(self, target: str):
        super().__init__(eval_type='starts_with', target=target)

    def __call__(self, input_value: str) -> EvalResult:
        return EvalResult(input_value.startswith(self.target), input_value)

class Webhook(Eval):
    def __init__(self, url: str):
        super().__init__(eval_type='webhook', target=url)

    def __call__(self, input_value: Any) -> EvalResult:
        try:
            response = requests.post(self.target, json={'input': input_value})
            result = response.json()
            return EvalResult(result.get('pass', False), result)
        except Exception as e:
            return EvalResult(False, str(e))

# Contains
# contains_eval = Contains("hello")
contains_eval = Eval('contains', target='hello')
print("Contains 'hello':", contains_eval("hello world"))
exit()

# ContainsAll
contains_all_eval = ContainsAll(["apple", "banana", "cherry"])
print("Contains all fruits:", contains_all_eval("I like apple, banana, and cherry."))

# ContainsAny
contains_any_eval = ContainsAny(["dog", "cat", "bird"])
print("Contains any pet:", contains_any_eval("I have a cat at home."))

# ContainsJson
contains_json_eval = ContainsJson(schema={"type": "object", "properties": {"name": {"type": "string"}}})
print("Contains valid JSON:", contains_json_eval('{"name": "John", "age": 30}'))

# ContainsSql
contains_sql_eval = ContainsSql()
print("Contains SQL:", contains_sql_eval("SELECT * FROM users WHERE age > 18;"))

# ContainsXml
contains_xml_eval = ContainsXml()
print("Contains XML:", contains_xml_eval("<root><item>value</item></root>"))

# Cost
cost_eval = Cost(0.01)
print("Cost below threshold:", cost_eval(0.005))

# Equals
equals_eval = Equals("exact match")
print("Exact match:", equals_eval("exact match"))

# IContains
icontains_eval = IContains("HELLO")
print("Case-insensitive contains:", icontains_eval("hello world"))

# IContainsAll
icontains_all_eval = IContainsAll(["APPLE", "BANANA", "CHERRY"])
print("Case-insensitive contains all:", icontains_all_eval("I like Apple, banana, and Cherry."))

# IContainsAny
icontains_any_eval = IContainsAny(["DOG", "CAT", "BIRD"])
print("Case-insensitive contains any:", icontains_any_eval("I have a cat at home."))

# IsJson
is_json_eval = IsJson(schema={"type": "object", "properties": {"name": {"type": "string"}}})
print("Is valid JSON:", is_json_eval('{"name": "John", "age": 30}'))

# IsSql
is_sql_eval = IsSql(allowed_statements=["SELECT", "INSERT", "UPDATE", "DELETE"])
print("Is valid SQL:", is_sql_eval("SELECT * FROM users WHERE age > 18;"))

# IsValidOpenAIFunctionCall
function_schema = {
    "name": "get_current_weather",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
}
is_valid_openai_function_call_eval = IsValidOpenAIFunctionCall(function_schema)
print("Is valid OpenAI function call:", is_valid_openai_function_call_eval({
    "name": "get_current_weather",
    "arguments": '{"location": "London", "unit": "celsius"}'
}))

# IsValidOpenAIToolsCall
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]
is_valid_openai_tools_call_eval = IsValidOpenAIToolsCall(tools_schema)
print("Is valid OpenAI tools call:", is_valid_openai_tools_call_eval([{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "arguments": '{"location": "London", "unit": "celsius"}'
    }
}]))

# IsXml
is_xml_eval = IsXml()
print("Is valid XML:", is_xml_eval("<root><item>value</item></root>"))

# Javascript
# Note: This would require a JavaScript runtime, so we'll skip the actual execution
javascript_eval = Javascript("function(input) { return input.length > 5; }")
print("JavaScript evaluator created:", javascript_eval)

# Latency
latency_eval = Latency(100)  # 100 ms threshold
print("Latency below threshold:", latency_eval(50))

# Levenshtein
levenshtein_eval = Levenshtein("kitten", 2)
print("Levenshtein distance within threshold:", levenshtein_eval("sitting"))

# PerplexityScore
perplexity_score_eval = PerplexityScore()
print("Perplexity score:", perplexity_score_eval(10.5))

# Perplexity
perplexity_eval = Perplexity(50)
print("Perplexity below threshold:", perplexity_eval(30))

# Python
def custom_python_function(input_value):
    return len(input_value) > 10

python_eval = Python(custom_python_function)
print("Python function evaluation:", python_eval("Hello, World!"))

# Regex
regex_eval = Regex(r'\d+')
print("Regex match:", regex_eval("There are 123 apples"))

# RougeN
rouge_eval = RougeN("The cat is on the mat", 1, 0.5)
print("ROUGE-N score above threshold:", rouge_eval("A cat sits on a mat"))

# StartsWith
starts_with_eval = StartsWith("Hello")
print("Starts with 'Hello':", starts_with_eval("Hello, World!"))

# Webhook
# Note: This would require an actual server to test, so we'll skip the execution
webhook_eval = Webhook("https://api.example.com/webhook")
print("Webhook evaluator created:", webhook_eval)

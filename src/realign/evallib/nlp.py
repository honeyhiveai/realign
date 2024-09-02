from realign.evaluators import evaluator
from realign.utils import try_import
from realign.llm_utils import OpenAIMessage


@evaluator
def word_count(text: str) -> int:
    """
    Count the number of words in the given text.
    
    Args:
        text (str): The input text to count words from.
    
    Returns:
        int: The number of words in the text.
    """
    return len(text.split())

@evaluator
def character_count(text: str) -> int:
    """
    Count the number of characters in the given text.
    
    Args:
        text (str): The input text to count characters from.
    
    Returns:
        int: The number of characters in the text.
    """
    return len(text)

@evaluator
def token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in the given text using tiktoken.
    
    Args:
        text (str): The input text to count tokens from.
        encoding_name (str): The name of the tiktoken encoding to use. Default is "cl100k_base".
    
    Returns:
        int: The number of tokens in the text.
    """
    tiktoken = try_import('tiktoken')
    
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)


@evaluator
def chat_word_count_by_role(messages: list[dict | OpenAIMessage], role: str | None = None) -> int | dict:
    """
    Count the number of words in the messages for a given role or all roles.
    
    Args:
        messages (list[dict | OpenAIMessage]): The list of messages.
        role (str, optional): The specific role to count words for. If None, count for all roles.
    
    Returns:
        int | dict: If role is specified, returns the word count for that role.
                    If role is None, returns a dictionary with word counts for all roles.
    """
    word_counts = {}
    
    for message in messages:
        if isinstance(message, dict):
            message_role = message.get('role')
            message_content = message.get('content', '')
        else:  # Assuming OpenAIMessage
            message_role = message.role
            message_content = message.content
        
        word_count = len(message_content.split())
        word_counts[message_role] = word_counts.get(message_role, 0) + word_count
    
    if role is not None:
        return word_counts.get(role, 0)
    
    return word_counts


@evaluator
def levenshtein_distance(str1: str, str2: str) -> int:
    Levenshtein = try_import('Levenshtein')
    if Levenshtein:
        return Levenshtein.distance(str1, str2)
    else:
        print("Levenshtein library not found. Please install it using 'pip install python-Levenshtein'")
        return -1  # Return a sentinel value to indicate failure

cosim = try_import('sklearn.metrics.pairwise', 'cosine_similarity')
np = try_import('numpy')

@evaluator
def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vec1 (list[float]): The first vector.
        vec2 (list[float]): The second vector.
    
    Returns:
        float: The cosine similarity between the two vectors.
    """
    if cosine_similarity is not None and np is not None:
        A = np.array([vec1])
        B = np.array([vec2])
        return cosine_similarity(A, B)[0][0]
    else:
        print("Required libraries (numpy and scikit-learn) not found. Please install them.")
        return -1  # Return a sentinel value to indicate failure


@evaluator
def keyword_assertion(text: str, keywords: list[str], case_sensitive: bool = False) -> bool:
    """
    Assert that all given keywords are present in the text.
    
    Args:
        text (str): The input text to check for keywords.
        keywords (list[str]): List of keywords to search for in the text.
        case_sensitive (bool): Whether the search should be case-sensitive. Default is False.
    
    Returns:
        bool: True if all keywords are present, False otherwise.
    """
    if not case_sensitive:
        text = text.lower()
        keywords = [keyword.lower() for keyword in keywords]
    
    return all(keyword in text for keyword in keywords)

# Run the test function
if __name__ == "__main__":
    def test_nlp_evaluators():
        """
        Test all the NLP evaluators in this module.
        
        Returns:
            dict: A dictionary containing the test results for each evaluator.
        """
        test_text = "This is a sample text for testing NLP evaluators."
        
        results = {}
        
        # Test word_count
        results['word_count'] = word_count(test_text)
        assert results['word_count'] == 9, f"Expected 9 words, but got {results['word_count']}"
        
        # Test character_count
        results['character_count'] = character_count(test_text)
        assert results['character_count'] == 49, f"Expected 49 characters, but got {results['character_count']}"
        
        # Test token_count
        results['token_count'] = token_count(test_text)
        assert results['token_count'] == 12, f"Expected 12 tokens, but got {results['token_count']}"
        
        print("NLP Evaluator Test Results:")
        for evaluator_name, result in results.items():
            print(f"{evaluator_name}: {result}")
            
        print("\nTesting chat_word_count_by_role:")
        test_messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm fine, thank you!"},
            {"role": "user", "content": "What is your name?"}
        ]
        results['chat_word_count_by_role'] = chat_word_count_by_role(test_messages)
        assert results['chat_word_count_by_role']['user'] == 8, f"Expected 7 words for user, but got {results['chat_word_count_by_role']['user']}"
        assert results['chat_word_count_by_role']['assistant'] == 4, f"Expected 4 words for assistant, but got {results['chat_word_count_by_role']['assistant']}"
        
        
        results['edit_distance'] = levenshtein_distance('test_tects', 'test_text')
        assert results['edit_distance'] == 2, f"Expected 2 edit distance, but got {results['edit_distance']}"
        
        print("All assertions passed successfully!")
        return results


    test_nlp_evaluators()


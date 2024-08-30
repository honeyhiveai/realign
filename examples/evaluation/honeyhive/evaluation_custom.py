import os
from openai import OpenAI
from pinecone import Pinecone
from honeyhive.tracer.custom import trace
import realign
from realign.evaluation import Evaluation
realign.tracing.honeyhive_key = '<YOUR_HONEYHIVE_API_KEY>'
realign.tracing.honeyhive_project = '<YOUR_HONEYHIVE_PROJECT_NAME>'

### Assuming OPENAI_API_KEY and PINECONE_API_KEY have been set in the user environment

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Pinecone setup
PINECONE_INDEX_NAME = "chunk-size-512"


# @trace -> allows custom spans and data to be sent to then evaluation
@trace(
    config={
        "app_version": "1.0.0",
        "source": "production"
    }
)
def get_pinecone_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(PINECONE_INDEX_NAME)

@trace(
    config={
        "model": "text-embedding-ada-002",
        "provider": "OpenAI",
        "template": "text-embedding"
    }
)
def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

@trace(
    config={
        "model": "gpt-4o",
        "provider": "OpenAI",
        "temperature": 0.7,
        "max_tokens": 150
    }
)
def generate_response(system_prompt, user_query):
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )
    return completion.choices[0].message.content

@trace(
    config={
        "type": "context_extraction",
        "source": "pinecone"
    }
)
def extract_context(search_results, user_query):
    context = []
    for match in search_results['matches']:
        if 'metadata' in match and 'text' in match['metadata']:
            context.append(match['metadata']['text'])
        elif 'metadata' in match:
            for key, value in match['metadata'].items():
                if isinstance(value, str) and len(value) > 50:
                    context.append(value)
                    break
    
    if not context:
        raise ValueError("No valid context found in search results")
    
    context_text = "\n\n".join(context)
    return context_text

@trace(
    config={
        "app": "search",
        "provider": "Pinecone",
        "instance": PINECONE_INDEX_NAME,
        "embedding_model": "text-embedding-ada-002",
        "chunk_size": 512,
        "chunk_overlap": 0
    }
)
def document_search(query):
    try:
        index = get_pinecone_index()
        
        # Get query embedding
        query_embedding = get_embedding(query)
        
        # Search Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Extract context from search results
        context_text = extract_context(search_results, query)
        
        system_prompt = f"""You are an AI assistant that helps with document searches. 
        Use the following context to answer the user's query. 
        If the information is not in the context, say so.
        
        Context:
        {context_text}
        """
        
        response = generate_response(system_prompt, query)
        
        return response
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(f"Error details: {error_message}")
        return error_message

@trace(
    config={
        "type": "evaluation",
        "source": "ramp_product_qa"
    }
)
def process_question(question):
    result = document_search(question)
    print(f"Query: {question}")
    print(f"Response: {result}")
    print("---")
    return result

class SampleRAGEvaluation(Evaluation):

    async def main(self, run_context):
        # inputs -> json as dict
        inputs = run_context.inputs
        
        ### Evaluation body starts here ###

        result = process_question(inputs['question'])

        ### Evaluation body ends here ###

        # Return output to stitch to the session
        return result


json_list = [
        {
            "question": "What are the key features of the Ramp product?"  
        },
        {
            "question": "How does Ramp help with expense management?"  
        },
        {
            "question": "What are the benefits of using Ramp for corporate cards?" 
        },
        {
            "question": "Can you explain Ramp's bill pay feature?" 
        },
        {
            "question": "How does Ramp's reporting functionality work?" 
        }  
    ]

evaluation = SampleRAGEvaluation(
    evaluation_name='Sample RAG Evaluation', 
    query_list=json_list

)
evaluation.run()

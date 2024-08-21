import asyncio
import time
import random
import os
import json
from typing import Optional

from litellm import acompletion, completion, token_counter
from litellm.exceptions import RateLimitError, APIConnectionError, BadRequestError
from litellm.utils import ModelResponse

class ModelRouter:
    
    RETRIABLE_EXCEPTIONS = [RateLimitError, APIConnectionError, Exception]

    def __init__(self, model, batch_size, requests_per_minute):
        # each model has its own router
        self.model = model

        # Request queue and request times
        self.request_queue = asyncio.Queue()
        self.request_times = asyncio.Queue()
        
        # Rate limiting parameters
        self.requests_per_minute = requests_per_minute
        self.rate_limit_interval = 60 # seconds
        self.last_request_time = time.time() - self.rate_limit_interval 

        # Continuous queue processing task
        self.processing_task = None
        self.processing_task_started = False
        
        # Batch size
        self.batch_size = batch_size
        self._batch_size = batch_size
        self.retry_in_last_batch = False
        
        # Exponential backoff parameters
        self.max_retries = 5
        self.base_delay = 1
        self.max_delay = 60
        self.base_multiple = 2
        
        # stats
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_retries = 0
        self.total_cost = 0.0
        self.total_duration = 0.0
        self.total_tokens = 0

    async def acompletion(self, **params) -> ModelResponse:
        
        # Create a future to store the result of the API call
        future = asyncio.Future()

        # Add the future and params to the request queue
        await self.request_queue.put((future, params))
        # print(f'{self.model} Queue size: {self.request_queue.qsize()}')
        
        return await future
    
    def completion(self, **params) -> Optional[ModelResponse]:
        
        for attempt in range(self.max_retries):
            try:
                self.total_requests += 1
                
                # start the timer
                start = time.time()

                # Make the LLM API call
                response: Optional[ModelResponse] = completion(**params)
                
                if response is None:
                    raise Exception(f'API call for {self.model} failed: {response}')
                
                # end the timer
                end = time.time()
                
                if attempt > 0:
                    print(f'API call for {self.model} succeeded after {attempt} retries!')
                    
                # update the stats
                self.successful_requests += 1
                self.total_retries += attempt
                self.total_duration += end - start
                if 'messages' in params:
                    self.total_tokens += token_counter(model=self.model, messages=params['messages'])
                self.total_cost += response._hidden_params.get('response_cost', 0)
                
                print(f'Successfully called {self.model} in {end - start:.3f} seconds.')
                return response
            except Exception as e:
                    self.raise_or_continue(attempt, e)
        
        
        return response

    async def _continuous_process_queue(self):
        print(f'{self.model} Processing requests...')
        while True:
            if not self.request_queue.empty():
                # print(f'{self.model} Processing queue.')
                await self._process_queue_in_batches()
            else:
                await asyncio.sleep(0.1)  # Small delay to prevent busy-waiting
                
    async def shutdown(self):
        # print(f'{self.model} model router shutting down.')
        if self.processing_task_started and self.processing_task is not None:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                # print("Processing task was cancelled successfully.")
                pass

    def get_stats(self):
        return {
            'Routing Stats': {
                'Total Requests': self.total_requests,
                'Successful Requests': self.successful_requests,
                'Failed Requests': self.failed_requests,
                'Total Retries': self.total_retries,
            },
            'Model Stats (successful requests)': {
                'Total Tokens': self.total_tokens,
                'Total Cost': "$ " + str(round(self.total_cost, 3)),
                'Average Tokens per Request': str(round((self.total_tokens / self.successful_requests) if self.successful_requests > 0 else 0, 3)) + " tokens",
                'Average Cost per Request': "$ " + str(round((self.total_cost / self.successful_requests) if self.successful_requests > 0 else 0, 3)),
                'Average Latency per Request': str(round((self.total_duration / self.successful_requests) if self.successful_requests > 0 else 0, 3)) + " sec",
            }
        }

    async def _process_queue_in_batches(self):
        # print(f'{self.model} Processing queue in batches.')
        # halve the batch size if a retry was made in the last batch
        if self.retry_in_last_batch:
            self._batch_size = max(self._batch_size // 2, 1)
            print(f'{self.model} Reduced batch size to {self._batch_size} due to rate limit errors.')

        self.retry_in_last_batch = False

        # Get a new batch of requests from the queue
        batch: list[tuple[asyncio.Future, dict]] = []
        for _ in range(min(self._batch_size, self.request_queue.qsize())):
            batch.append(await self.request_queue.get())
        
        # Wait for the rate limit to be reset if necessary
        await self._wait_for_rate_limit(len(batch))
        
        # Update the request times
        await self._update_request_times(len(batch), time.time())

        # Make API calls for the batch
        results: list[ModelResponse | Exception] = await asyncio.gather(
            *[
                self.make_api_call(**params) 
                for _, params in batch
            ],
            return_exceptions=True
        )
        
        # Set the results for each future in the batch
        for (future, _), result in zip(batch, results):
            future.set_result(result)
            self.request_queue.task_done()
        
        # print(f'{self.model} Processed batch. Queue size: {self.request_queue.qsize()}')

    async def _wait_for_rate_limit(self, batch_size):
        now = time.time()
        while self.request_times.qsize() + batch_size > self.requests_per_minute:
            oldest_request = await self.request_times.get()
            time_since_oldest = now - oldest_request
            if time_since_oldest < self.rate_limit_interval:
                wait_time = self.rate_limit_interval - time_since_oldest
                print(f'Rate limit of {self.requests_per_minute} RPM reached for {self.model}. Retrying in {wait_time:.3f} seconds. To increase the rate limit, increase the requests_per_minute parameter in the model router settings.')
                await asyncio.sleep(wait_time)
            now = time.time()

    async def _update_request_times(self, batch_size, batch_request_time):
        for _ in range(batch_size):
            if self.request_times.qsize() >= self.requests_per_minute:
                await self.request_times.get()
            await self.request_times.put(batch_request_time)
        self.last_request_time = batch_request_time
        
    def calculate_delay(self, attempt):
        """Calculate delay with exponential backoff."""
        delay = min(self.base_delay * (self.base_multiple ** attempt), self.max_delay)
        return delay
    
    def raise_or_continue(self, attempt, e):
        if e not in self.RETRIABLE_EXCEPTIONS:
            self.failed_requests += 1
            self.total_retries += attempt
            print(f'API call for model {self.model} failed. Exception of class {e.__class__} is not retriable: {str(e)}')
            raise # Re-raise on non-retriable exceptions
        elif attempt == self.max_retries - 1:
            self.failed_requests += 1
            self.total_retries += attempt
            print(f'API call for model {self.model} failed after {attempt + 1} retries: {str(e)}')
            raise  # Re-raise on last attempt
        else:
            # continue and retry
            pass
    
    async def make_api_call(self, **params) -> Optional[ModelResponse]:
        
        for attempt in range(self.max_retries):
            try:
                self.total_requests += 1
                
                # start the timer
                start = asyncio.get_event_loop().time()

                # Make the LLM API call
                response: Optional[ModelResponse] = await acompletion(**params)
                
                if response is None:
                    raise Exception(f'API call for {self.model} failed: {response}')
                
                # end the timer
                end = asyncio.get_event_loop().time()
                
                if attempt > 0:
                    print(f'API call for {self.model} succeeded after {attempt} retries!')
                    
                # update the stats
                self.successful_requests += 1
                self.total_retries += attempt
                self.total_duration += end - start
                if 'messages' in params:
                    self.total_tokens += token_counter(model=self.model, messages=params['messages'])
                self.total_cost += response._hidden_params.get('response_cost', 0)
                
                print(f'Successfully called {self.model} in {end - start:.3f} seconds.')
                return response
            
            except BadRequestError as e:
                # BadRequestError is not retriable
                self.raise_or_continue(attempt, e)
                
            except RateLimitError as e:
                self.raise_or_continue(attempt, e)
                
                # Get the retry-after header from the response, or use the calculated delay
                delay = float(e.response.headers.get('retry-after', self.calculate_delay(attempt)))
                delay += random.uniform(0, 0.1 * self._batch_size) # Add jitter
                
                # Set the retry in the last batch                
                self.retry_in_last_batch = True                

                print(f'RateLimitError encountered for {self.model}. Retrying after {delay:.3f} seconds (Attempt {attempt + 1}/{self.max_retries}). Please increase the requests_per_minute parameter in the model router settings to throttle the rate limit.\n')
                await asyncio.sleep(delay)
                
            except APIConnectionError as e:
                self.raise_or_continue(attempt, e)

                delay = self.calculate_delay(attempt) + random.uniform(0, 0.1 * self._batch_size) # Add jitter
                
                self.retry_in_last_batch = True
                
                # print the error message
                print(f'APIConnectionError encountered for {self.model}. Retrying after {delay:.3f} seconds (Attempt {attempt + 1}/{self.max_retries})\n')
                await asyncio.sleep(delay)
                
            except Exception as e:
                self.raise_or_continue(attempt, e)
                
                delay = self.calculate_delay(attempt) + random.uniform(0, 0.1 * self._batch_size) # Add jitter
                
                self.retry_in_last_batch = True
                                
                # print the error message
                print('Exception encountered:', e)
                print(f'\nAPI call for model {self.model} failed. Retrying after {delay} seconds (Attempt {attempt + 1}/{self.max_retries})\n')
                await asyncio.sleep(delay)
        
        # This line should never be reached due to the re-raise above, but including for completeness
        raise Exception(f"API call failed after {self.max_retries} retries. Please reach out to our team.")

class Router:
    
    DEFAULT_SETTINGS = {
        '*/*': {
            'batch_size': 1000,
            'requests_per_minute': 10000
        },
        'groq/*': {
            'batch_size': 5,
            'requests_per_minute': 30
        },
        'openai/*': {
            'batch_size': 100,
            'requests_per_minute': 500
        }
    }

    def __init__(self, model_router_settings: dict = None):

        # load model router settings
        self.model_router_settings = model_router_settings
        
        # router for each model
        self.model_routers: dict[str, ModelRouter] = dict()
    
    async def shutdown(self):
        for model in self.model_routers:
            await self.model_routers[model].shutdown()
    
    # def __del__(self):
        
    #     # set event loop if not set
    #     if not (loop := asyncio.get_event_loop()):
    #         asyncio.set_event_loop(asyncio.new_event_loop())
    #     else:
    #         asyncio.set_event_loop(loop)

    #     loop.run_until_complete(self.shutdown())
            
    def get_stats(self):
        stats = dict()
        for model in self.model_routers:
            stats[model] = self.model_routers[model].get_stats()
        return stats
        
    def resolve_model_router_settings(self, model: str):
        # Load model router settings
        if self.model_router_settings is not None:
            pass
        elif (router_envos := os.getenv('MODEL_ROUTER_SETTINGS')):
            self.model_router_settings = json.loads(router_envos)
        else:
            self.model_router_settings = Router.DEFAULT_SETTINGS
        
        # set the wildcard model router settings to the default max
        if '*/*' in self.model_router_settings and self.model_router_settings['*/*'] == '*':
            self.model_router_settings['*/*'] = Router.DEFAULT_SETTINGS['*/*']
        
        # TODO: Better routing
        if model in self.model_router_settings:
            settings = self.model_router_settings[model]
        elif (provider := model.split('/')[0]) + '/*' in self.model_router_settings:
            settings = self.model_router_settings[provider + '/*']
        else:
            settings = self.model_router_settings['*/*']
        # print('Using settings:', settings, 'for model', model)
        return settings
    
    def create_router_if_new_model(self, model: str):
        # Create a model router if it doesn't exist
        if model not in self.model_routers:
            settings = self.resolve_model_router_settings(model)
            self.model_routers[model] = ModelRouter(model, 
                                                    settings['batch_size'], 
                                                    settings['requests_per_minute'])
            print(f'Created model router for {model}')
    
    async def acompletion(self, **params) -> ModelResponse:
        
        assert 'model' in params, 'Model must be provided in the params'
        
        model: str = params['model']
        self.create_router_if_new_model(model)

        model_router: ModelRouter = self.model_routers[model]
        if not model_router.processing_task_started:
            asyncio.create_task(model_router._continuous_process_queue())
            model_router.processing_task_started = True
        
        # Make the Chat Completion call
        litellm_response: ModelResponse = await model_router.acompletion(**params)

        return litellm_response
    
    def completion(self, **params) -> ModelResponse:
        assert 'model' in params, 'Model must be provided in the params'
        
        model: str = params['model']
        self.create_router_if_new_model(model)

        model_router: ModelRouter = self.model_routers[model]
        
        # Make the Chat Completion call
        litellm_response: ModelResponse = model_router.completion(**params)

        return litellm_response

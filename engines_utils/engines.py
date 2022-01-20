from more_itertools import chunked
import asyncio
from tenacity import TryAgain, retry, stop_after_delay, wait_random_exponential, wait_fixed
import aiohttp
import math
import traceback


"""
Helper module to make async bulk calls to Primer.ai Engines.

Making asynchronous requests to the Primer.ai Engines speeds up the document throughput in your application.
Using an asynchronous design pattern, 'waiting time' while expecting a result from the API servers can be used
productively by carrying out other operations in the program, for example to trigger other concurrent requests, or
process the results from previous requests. See https://realpython.com/async-io-python/ for an excellent overview.

This module follows a queue design (https://realpython.com/async-io-python/#using-a-queue) to be able to process large
document sets in a reasonable time. The queue allows creating N consumers, each making independent asynchronous requests
to the API server. Additionally, the functions below also allow batching of documents, so that a single request can
return results on multiple documents. At any given, N_consumers x batch_size document will be being processed.

These helper functions allow implementing Asynchronous Processing with Batching as described here
https://developers.primer.ai/docs#asynchronous-processing-with-batching, including repeated polling to check if results
are available.

From a notebook, all that is required is:

```
documents = [{"id": 123, "text": "lorem ipsum"}, ... ]
results = infer_model_on_docs(documents, model_name='abstractive_topics', api_key=XXXX)
# results will be a dictionary or document id to topics list.
```

From inside a script, use:

```
import asyncio
documents = [{"id": 123, "text": "lorem imsum"}, ... ]
results = asyncio.run(infer_model_on_docs(documents, model_name='abstractive_topics', api_key=XXXX))
```

"""


ENGINES_BASE_URL = "https://engines.primer.ai/api"
MODEL_API_ROUTES = {"abstractive_topics": "/v1/generate/abstractive_topics",
                    "key_phrases": "/v1/generate/phrases",
                    "web_scraper": "/v1/extract/scrape"}
RESULT_API_ROUTE = "/v1/result/"



async def infer_model_on_docs(docs, model_name, api_key, q_length=10, batch_size=10, **kwargs):
    """

    :param docs: list[dict] : [{"id": 123, "text": "lorem imsum"}, ... ]
    :param model_name: str : a key in MODEL_API_ROUTES
    :param api_key: str: Get one at https://developers.primer.ai/
    :param q_length: number of consumers
    :param batch_size: number of documents per request
    :param kwargs: dict : additional parameters to include in request payload
    :return: dict of document id to request results.
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_key}",
    }

    timeout = aiohttp.ClientTimeout(total=200)
    async with aiohttp.ClientSession(timeout=timeout, headers=headers, raise_for_status=False) as session:
        inferences = await process_doc_list(session, docs, model_name, batch_size, q_length, **kwargs)

    return inferences


async def process_doc_list(session, docs, model_name, batch_size, ncon: int, **kwargs):
    # See https://realpython.com/async-io-python/#using-a-queue
    results = {}
    q = asyncio.Queue(maxsize=ncon)
    producers = [asyncio.ensure_future(produce(docs, batch_size, q))]
    consumers = [asyncio.ensure_future(consume(session, q, model_name, results, **kwargs)) for n in range(ncon)]
    await asyncio.gather(*producers)
    await q.join()  # Implicitly awaits consumers, too
    for c in consumers:
        c.cancel()
    return results


async def produce(docs, batch_size, q: asyncio.Queue) -> None:
    # Group documents in batch_size and add them to queue
    for i, batch in enumerate(chunked(docs, batch_size)):
        await q.put(batch)
        if i % 10 == 0:
            print(f"Producer added doc {i} to queue.")
    print("Producer is done!")


async def consume(session, q, model_name, results, **kwargs) -> None:
    while True:
        batch = await q.get()
        try:
            if len(batch) == 1:
                # If not batching, make single document call to API
                doc_response = await make_call(session, batch, model_name, **kwargs)
                results[batch[0]["id"]] = doc_response
            else:
                #
                response_list = await model_batch(session, batch, model_name, **kwargs)
                for doc_response, doc in zip(response_list, batch):
                    results[doc["id"]] = doc_response
            q.task_done()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            q.task_done()
            continue


async def model_batch(session, docs, model_name, sleep=5, **kwargs):
    # Log a batch request with API server
    task_id = await make_call(session, docs, model_name, **kwargs)

    # Impose a non-blocking sleep. No point calling wait_for_task immediately for demanding models
    await asyncio.sleep(sleep)

    # Check if results are available using task_id
    task = await wait_for_task(session, task_id)

    return task


@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_delay(300),
    reraise=True,
)
async def make_call(session, docs, model_name, **kwargs):
    # Prepare request payload
    texts = [d["text"] for d in docs]
    if len(texts)==1:
        # Just a string if single-doc
        texts = texts[0]
    input_data = {"text": texts}
    # Add optional arguments
    input_data.update(kwargs)

    # Make call
    url = ENGINES_BASE_URL + MODEL_API_ROUTES[model_name]
    async with session.post(url, json=input_data) as resp:
        task_response = await resp.json(content_type=None)

    if resp.status == 200:
        # Batch requests return a task_id for subsequent polling
        # Single doc requests return results directly
        # See https://developers.primer.ai/docs#asynchronous-processing-with-batching
        return task_response

    if resp.status == 400:
        print(f"400: {resp}, {input_data}")
        asyncio.sleep(6)
        raise TryAgain

    if resp.status == 429:
        s = math.ceil(float(resp.headers['Retry-After']))
        print(f"Rate limit hit: waiting {s} seconds...")
        asyncio.sleep(s)
        raise TryAgain

    # Catch anything else
    raise TryAgain


@retry(
    wait=wait_fixed(30),
    stop=stop_after_delay(300),
    reraise=True,
)
async def wait_for_task(session, name):
    # Ping task-specific results endpoint
    url = ENGINES_BASE_URL + RESULT_API_ROUTE + name
    async with session.get(url) as resp:
        task = await resp.json(content_type=None)

    if resp.status==202:
        print(f"Task {task} still being processed, will retry...")
        raise TryAgain

    if resp.status==429:
        s = math.ceil(float(resp.headers['Retry-After']))
        print(f"Rate limit hit: waiting {s} seconds...")
        asyncio.sleep(s)
        raise TryAgain

    return task










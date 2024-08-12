import json
import logging
import os
from functools import wraps
from time import perf_counter, sleep
from typing import Any, Callable, TypedDict

import google.generativeai as genai  # type: ignore
from google.generativeai.types import (  # type: ignore
    GenerateContentResponse,
    HarmBlockThreshold,
    HarmCategory,
)

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", force=True
)

logger = logging.getLogger("gemini")


class Response(TypedDict):
    """
    Typed dictionary representing the response from the Gemini model.

    Parameters
    ----------
    model : str
        The name of the model used for generation.
    completion : str
        The generated text completion.
    parsed_completion : None | dict[str, Any] | list[dict[str, Any]]
        The parsed completion, if applicable (e.g., JSON).
        Will be `None` if the completion could not be parsed as JSON.
    usage_input_tokens : int
        Number of input tokens used.
    usage_output_tokens : int
        Number of output tokens generated.
    usage_total_tokens : int
        Total tokens used in the request.
    duration_s : None | float
        Duration of the request in seconds. This is measured only for successful requests.
    """

    model: str
    completion: str
    parsed_completion: None | dict[str, Any] | list[dict[str, Any]]
    usage_input_tokens: int
    usage_output_tokens: int
    usage_total_tokens: int
    duration_s: None | float


class RateLimiter:
    """
    A class to handle rate limiting for API calls.

    This class ensures that the API call rate stays within the defined limits,
    preventing overuse and potential service disruptions.

    Parameters
    ----------
    max_requests_per_minute : int
        Maximum number of requests allowed per minute.
    max_tokens_per_minute : int
        Maximum number of tokens allowed per minute.
    max_requests_per_day : int
        Maximum number of requests allowed per day.
    timeout : int
        Time (in seconds) to wait when rate limit is reached. After the timeout,
        the request and token counts are reset.

    Attributes
    ----------
    max_requests_per_minute : int
        Maximum number of requests allowed per minute.
    max_tokens_per_minute : int
        Maximum number of tokens allowed per minute.
    max_requests_per_day : int
        Maximum number of requests allowed per day.
    timeout : int
        Time (in seconds) to wait when rate limit is reached.
    request_count : int
        Current number of requests made within the current minute.
    token_count : int
        Current number of tokens used within the current minute.
    daily_request_count : int
        Current number of requests made today.
    """

    def __init__(
        self,
        max_requests_per_minute: int,
        max_tokens_per_minute: int,
        max_requests_per_day: int,
        timeout: int,
    ):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_day = max_requests_per_day
        self.timeout = timeout
        self.request_count = 0
        self.token_count = 0
        self.daily_request_count = 0

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to apply rate limiting to a function.

        This decorator will enforce the rate limits for requests and tokens.
        If the rate limit is reached, it will wait for the specified timeout
        period and then reset the request and token counts.

        Parameters
        ----------
        func : Callable
            The function to be decorated.

        Returns
        -------
        Callable
            The decorated function.
        """

        @wraps(func)
        def wrapper(*args, **kwargs) -> Response:
            if self.daily_request_count >= self.max_requests_per_day:
                raise RuntimeError(
                    "Daily request limit reached. Please try again tomorrow."
                )
            if (
                self.request_count >= self.max_requests_per_minute
                or self.token_count >= self.max_tokens_per_minute
            ):
                logger.warning("Rate limit reached. Waiting for timeout...")
                sleep(self.timeout)
                self.request_count = 0
                self.token_count = 0
            result = func(*args, **kwargs)
            self.request_count += 1
            self.daily_request_count += 1
            return result

        return wrapper

    def update_token_count(self, tokens: int) -> None:
        """
        Update the token count for the current request.

        Parameters
        ----------
        tokens : int
            The number of tokens used in the request.
        """
        self.token_count += tokens


class RateLimitedGenerativeModel:
    """
    A class for interacting with the Google Gemini Flash model for text annotation.

    This class handles API configuration, rate limiting, and response formatting.

    Parameters
    ----------
    api_key : str
        Your Google Cloud API key.
    system_instruction : str
        System-level instructions or prompts for the model.
    model_name : str, optional
        The name of the Gemini model to use, defaults to "gemini-1.5-flash".
    temperature : float, optional
        Controls the randomness of the generated text, defaults to 1.
    top_p : float, optional
        Nucleus sampling parameter, defaults to 0.95.
    top_k : int, optional
        Top-k sampling parameter, defaults to 64.
    max_output_tokens : int, optional
        Maximum number of tokens to generate, defaults to 8192.
    timeout : int, optional
        Timeout for rate limiting (in seconds), defaults to 60.
    max_requests_per_minute : int, optional
        Maximum requests per minute, defaults to 15.
    max_tokens_per_minute : int, optional
        Maximum tokens per minute, defaults to 1_000_000.
    max_requests_per_day : int, optional
        Maximum requests per day, defaults to 1500.

    Attributes
    ----------
    generation_config : dict
        Configuration for text generation, including parameters such as
        temperature, top_p, top_k, and max_output_tokens.
    model : google.generativeai.GenerativeModel
        The Gemini model instance, initialized with the specified model_name,
        generation_config, and system instructions.
    rate_limiter : RateLimiter
        The rate limiter instance, used to control API usage according to
        specified limits.
    """

    def __init__(
        self,
        api_key: str,
        system_instruction: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 1,
        top_p: float = 0.95,
        top_k: int = 64,
        max_output_tokens: int = 8192,
        timeout: int = 60,
        max_requests_per_minute: int = 15,
        max_tokens_per_minute: int = 1_000_000,
        max_requests_per_day: int = 1500,
    ):
        genai.configure(api_key=api_key)

        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "application/json",
        }

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            },
            system_instruction=system_instruction,
        )

        self.rate_limiter = RateLimiter(
            max_requests_per_minute,
            max_tokens_per_minute,
            max_requests_per_day,
            timeout,
        )

    @property
    def requests_available(self) -> str:
        """
        Return a string indicating the remaining requests for the day.

        This method provides a status report on the number of requests
        made and the number of requests remaining for the day.

        Returns
        -------
        str
            A string indicating the current request count and remaining requests for the day.
        """
        remaining_requests = (
            self.rate_limiter.max_requests_per_day
            - self.rate_limiter.daily_request_count
        )
        return (
            f"Current request count: {self.rate_limiter.daily_request_count}, "
            f"remaining today: {remaining_requests}"
        )

    def generate_content(self, text: str) -> Response:
        """
        Generate content from the Gemini model with applied rate limiting.

        This method generates content based on the input text, while enforcing
        rate limits on the number of requests and tokens used.

        Parameters
        ----------
        text : str
            The input text for the model.

        Returns
        -------
        Response
            A dictionary containing the model's response, usage information,
            and the duration of the request.
        """

        @self.rate_limiter
        def _generate_content(text: str) -> Response:
            try:
                start = perf_counter()
                response = self.model.generate_content(text)
                end = perf_counter() - start
                formatted_response = self._format_response(response)
                formatted_response["duration_s"] = end
                return formatted_response
            except Exception as e:
                logger.exception(f"Error generating content: {e}")
                raise

        return _generate_content(text)

    def _format_response(self, response: GenerateContentResponse) -> Response:
        """
        Format the raw response from the Gemini model into a structured dictionary.

        This method attempts to parse the raw text response into a JSON object.
        If parsing fails, `parsed_completion` will be set to `None`. Additionally,
        this method updates the token count in the rate limiter based on the usage
        metadata provided by the model.

        Parameters
        ----------
        response : GenerateContentResponse
            The raw GenerateContentResponse object.

        Returns
        -------
        Response
            A formatted dictionary containing the model's response, parsed completion,
            and usage details including the number of input and output tokens, and the
            total tokens used.
        """
        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding response: {response.text}")
            logger.exception(e)
            parsed = None

        token_usage = {
            "input": response.usage_metadata.prompt_token_count,
            "output": response.usage_metadata.candidates_token_count,
            "total": response.usage_metadata.total_token_count,
        }
        self.rate_limiter.update_token_count(
            token_usage["total"] + self.generation_config["max_output_tokens"]
        )

        return Response(
            model=self.model.model_name,
            completion=response.text,
            parsed_completion=parsed,
            usage_input_tokens=token_usage["input"],
            usage_output_tokens=token_usage["output"],
            usage_total_tokens=token_usage["total"],
            duration_s=None,
        )


if __name__ == "__main__":
    system_instruction = ""
    text = ""

    extraction_model = RateLimitedGenerativeModel(
        api_key=os.environ["GEMINI_API_KEY"],
        system_instruction=system_instruction,
    )
    completion = extraction_model.generate_content(text)

    print(completion)
    print(extraction_model.requests_available)

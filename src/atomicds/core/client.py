from __future__ import annotations

import itertools
import os
import platform
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from importlib.metadata import version
from typing import Any, Callable  # type: ignore[ruleName]
from urllib.parse import urljoin

from requests import Session
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

__version__ = version("atomicds")


class BaseClient:
    """Base API client implementation"""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
    ):
        """
        Args:
            api_key (str | None): API key.
            endpoint (str): Root API endpoint.
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self._session = None

    @property
    def session(self) -> Session:
        """Session under which HTTP requests are issued"""
        if not self._session:
            self._session = self._create_session(self.api_key)
        return self._session  # type: ignore[return-value]

    def _get(
        self,
        sub_url: str,
        params: dict[str, Any] | None = None,
        deserialize: bool = True,
        base_override: str | None = None,
    ) -> list[dict[Any, Any]] | dict[Any, Any] | bytes | None:
        """Method for issuing a GET request

        Args:
            sub_url (str): API sub-url to use.
            params (dict[str, Any] | None): Params to pass in the GET request. Defaults to None.
            deserialize (bool): Whether to JSON deserialize the response data or return raw bytes. Defaults to True.
            base_overrise (str): Base URL to use instead of the default ADS API root URL.

        Raises:
            ClientError: If the response code returned is not within the range of 200-400.

        Returns:
            (list[dict] | dict | bytes | None): Deserialized JSON data or raw bytes. Returns None if response is a 404.

        """
        base_url = base_override or self.endpoint
        response = self.session.get(
            url=urljoin(base_url, sub_url), verify=True, params=params
        )
        if not response.ok:
            if response.status_code == 404:
                return None

            raise ClientError(
                f"Problem retrieving data from {sub_url} with parameters {params}. HTTP Error {response.status_code}: {response.text}"
            )
        if len(response.content) == 0:
            return None

        return response.json() if deserialize else response.content

    def _multi_thread(
        self,
        func: Callable[..., Any],
        kwargs_list: list[dict[str, Any]],
        progress_bar: tqdm | None = None,  # type: ignore[ruleName]
    ) -> list[Any]:
        """Handles running a function concurrently with a ThreadPoolExecutor

        Arguments:
            func (Callable): Function to run concurrently
            kwargs_list (list): List of keyword argument inputs for the function
            progress_bar (tqdm): Progress bar to show. Defaults to None.

        Returns:
            (list[Any]): List of results from passed function in the order of parameters passed
        """
        return_dict = {}

        kwargs_gen = iter(kwargs_list)

        ind = 0
        num_parallel = min(os.cpu_count() or 8, 8)
        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            # Get list of initial futures defined by max number of parallel requests
            futures = set()

            for kwargs in itertools.islice(kwargs_gen, num_parallel):
                future = executor.submit(
                    func,
                    **kwargs,
                )

                future.ind = ind  # type: ignore  # noqa: PGH003
                futures.add(future)
                ind += 1

            while futures:
                # Wait for at least one future to complete and process finished
                finished, futures = wait(futures, return_when=FIRST_COMPLETED)

                for future in finished:
                    data = future.result()

                    if progress_bar is not None:
                        progress_bar.update(1)

                    return_dict[future.ind] = data  # type: ignore  # noqa: PGH003

                # Populate more futures to replace finished
                for kwargs in itertools.islice(kwargs_gen, len(finished)):
                    new_future = executor.submit(
                        func,
                        **kwargs,
                    )

                    new_future.ind = ind  # type: ignore  # noqa: PGH003
                    futures.add(new_future)
                    ind += 1

        return [t[1] for t in sorted(return_dict.items())]

    @staticmethod
    def _create_session(api_key: str):
        """Create a requests session

        Args:
            api_key (str): API key to include in the header.

        Returns:
            (Session): Requests Session object

        """
        session = Session()
        session.headers = {"X-API-KEY": api_key}

        # User agent information
        atomicds_info = "atomicds/" + __version__
        python_info = f"Python/{sys.version.split()[0]}"
        platform_info = f"{platform.system()}/{platform.release()}"
        session.headers[
            "user-agent"
        ] = f"{atomicds_info} ({python_info} {platform_info})"

        # TODO: Add retry setting to configuration somewhere
        max_retry_num = 3
        retry = Retry(
            total=max_retry_num,
            read=max_retry_num,
            connect=max_retry_num,
            respect_retry_after_header=True,
            status_forcelist=[429, 504, 502],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session


class ClientError(Exception):
    """Generic error thrown by the Atomic Data Sciences API client"""

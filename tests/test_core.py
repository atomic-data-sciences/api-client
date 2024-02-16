import os
import pytest
from requests import Session

from atomicds.core import BaseClient
from unittest import mock

from atomicds.core.client import ClientError

@pytest.fixture
def base_client():
    api_key = os.environ.get("ADS_API_KEY", "")
    endpoint = os.environ.get("ADS_API_ENDPOINT", "") 

    return BaseClient(api_key=api_key, endpoint=endpoint)


def test_core_get_session(base_client):
    assert isinstance(base_client.session, Session)

def test_core_get_ok(base_client):

    heartbeat_response = base_client._get(sub_url="heartbeat/")
    assert heartbeat_response.get("status") == "OK"
    assert "time" in heartbeat_response
    assert "version" in heartbeat_response

    # Test other kwargs 
    heartbeat_response_raw = base_client._get(sub_url="heartbeat/", deserialize=False)
    assert isinstance(heartbeat_response_raw, bytes)

    heartbeat_response_raw = heartbeat_response_raw.decode("utf-8")
    assert "OK" in heartbeat_response_raw
    assert "time" in heartbeat_response
    assert "version" in heartbeat_response
   
    with mock.patch("requests.Session.get") as get:
        heartbeat_response = base_client._get(sub_url="", params={"test": "test"}, base_override="test")
        _, kwargs = get.call_args
        assert kwargs == {"url": "test", "params": {"test": "test"}, "verify": True}
        

def test_core_get_not_ok(base_client):
    
    # Mocked response object
    class Response:
        def __init__(self, ok, status_code):
            self.ok = ok
            self.status_code = status_code

    # 404
    with mock.patch("requests.Session.get", return_value=Response(False, 404)) as get:
        heartbeat_response = base_client._get(sub_url="", params={"test": "test"}, base_override="test")
        assert heartbeat_response is None 
        assert get.called

    # 400+
    with pytest.raises(ClientError, match="Problem retrieving data from"):
        with mock.patch("requests.Session.get", return_value=Response(False, 500)) as get:
            heartbeat_response = base_client._get(sub_url="", params={"test": "test"}, base_override="test")
            assert get.called

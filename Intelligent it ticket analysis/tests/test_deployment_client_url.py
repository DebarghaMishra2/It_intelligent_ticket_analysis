from frontend.api_client import ITSentinelClient


def test_client_adds_http_scheme_for_private_hostport():
    client = ITSentinelClient("it-sentinel-api:10000")
    assert client.base_url == "http://it-sentinel-api:10000"


def test_client_preserves_https_url():
    client = ITSentinelClient("https://api.example.onrender.com/")
    assert client.base_url == "https://api.example.onrender.com"

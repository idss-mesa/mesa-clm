"""assert_loopback (the D16 gate) and the CircuitBreaker, ported from mesa-anyjev's gateway
tests and extended for allow_remote + https."""

from __future__ import annotations

import threading

import pytest

from mesa_clm.net import (
    LOOPBACK_HOSTS,
    BreakerOpenError,
    CircuitBreaker,
    EndpointError,
    assert_loopback,
    is_loopback_host,
    redact_url,
)

# -- assert_loopback ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]", "LOCALHOST"])
@pytest.mark.parametrize("suffix", ["", ":8700", ":8090/", ":8000/prefix/"])
def test_loopback_matrix(scheme: str, host: str, suffix: str) -> None:
    url = f"{scheme}://{host}{suffix}"
    assert assert_loopback(url) == url.rstrip("/")
    assert assert_loopback(url, allow_remote=True) == url.rstrip("/")


def test_loopback_only_by_default() -> None:
    assert assert_loopback("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"
    with pytest.raises(EndpointError, match="loopback"):
        assert_loopback("http://192.0.2.10:8000")
    with pytest.raises(EndpointError, match="loopback"):
        assert_loopback("https://clm.example.org")
    with pytest.raises(EndpointError, match="/v1"):
        assert_loopback("http://127.0.0.1:8000/v1")
    with pytest.raises(EndpointError, match="/v1"):
        assert_loopback("http://127.0.0.1:8700/v1/")
    assert {"127.0.0.1", "localhost", "::1"} == LOOPBACK_HOSTS
    assert is_loopback_host("localhost") and not is_loopback_host("127.0.0.2")


def test_remote_needs_allow_remote_and_https() -> None:
    assert (
        assert_loopback("https://clm.example.org/", allow_remote=True) == "https://clm.example.org"
    )
    assert (
        assert_loopback("https://192.0.2.10:8700", allow_remote=True) == "https://192.0.2.10:8700"
    )
    with pytest.raises(EndpointError, match="https") as info:
        assert_loopback("http://clm.example.org:8700", allow_remote=True)
    assert "ssh -L" in str(info.value)
    with pytest.raises(EndpointError, match="ALLOW_REMOTE") as info:
        assert_loopback("https://clm.example.org", allow_remote=False)
    assert "ssh -L" in str(info.value)


def test_malformed_urls_are_refused() -> None:
    with pytest.raises(EndpointError, match="http"):
        assert_loopback("ftp://127.0.0.1")
    with pytest.raises(EndpointError, match="http"):
        assert_loopback("127.0.0.1:8700")
    with pytest.raises(EndpointError, match="no host"):
        assert_loopback("http:///path")
    with pytest.raises(EndpointError, match="not a valid URL"):
        assert_loopback("http://127.0.0.1:notaport")
    with pytest.raises(EndpointError, match="query"):
        assert_loopback("http://127.0.0.1:8700/?x=1")
    with pytest.raises(EndpointError, match="query"):
        assert_loopback("http://127.0.0.1:8700/#frag")
    with pytest.raises(EndpointError, match="http"):
        assert_loopback("")


def test_credentials_in_the_url_are_refused_and_not_echoed() -> None:
    with pytest.raises(EndpointError, match="credentials") as info:
        assert_loopback("http://user:sekrit@127.0.0.1:8700")
    assert (
        "sekrit" not in str(info.value) and "user" not in str(info.value).split("carries")[0][40:]
    )
    with pytest.raises(EndpointError) as info:
        assert_loopback("https://svc:sekrit@clm.example.org", allow_remote=True)
    assert "sekrit" not in str(info.value)


def test_what_names_the_endpoint() -> None:
    with pytest.raises(EndpointError, match=r"^encoder must be reached"):
        assert_loopback("http://10.0.0.1:8090", what="encoder")


def test_redact_url() -> None:
    assert redact_url("http://u:p@127.0.0.1:8700/x?y=1#z") == "http://127.0.0.1:8700/x"
    assert redact_url("https://[::1]:8700/") == "https://[::1]:8700/"
    assert redact_url("http://localhost") == "http://localhost"
    assert redact_url("http://h:bad") == "<unparseable URL>"


# -- CircuitBreaker ---------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_breaker_opens_after_failures_and_closes_after_open_for() -> None:
    clock = _Clock()
    br = CircuitBreaker(failures=3, open_for=60, name="clm-serve", clock=clock)
    br.check()
    br.failure()
    br.failure()
    assert not br.is_open and br.consecutive_failures == 2
    br.check()  # still closed
    br.failure()
    assert br.is_open
    with pytest.raises(BreakerOpenError, match="clm-serve circuit breaker is open") as info:
        br.check()
    assert isinstance(info.value, EndpointError)
    assert "60s" in str(info.value) and "3 consecutive failures" in str(info.value)
    clock.now += 59.9
    with pytest.raises(BreakerOpenError):
        br.check()
    clock.now += 0.2
    br.check()  # cool-down over: closed, count reset
    assert not br.is_open and br.consecutive_failures == 0


def test_success_resets_the_count() -> None:
    br = CircuitBreaker(failures=2, open_for=60)
    br.failure()
    br.success()
    br.failure()
    assert not br.is_open
    br.failure()
    assert br.is_open
    br.reset()
    assert not br.is_open and br.consecutive_failures == 0
    br.check()


def test_defaults_and_validation() -> None:
    br = CircuitBreaker()
    assert (br.failures, br.open_for, br.name) == (3, 60.0, "endpoint")
    assert "_lock" not in repr(br) and "clock" not in repr(br)
    with pytest.raises(ValueError, match="failures"):
        CircuitBreaker(failures=0)
    with pytest.raises(ValueError, match="open_for"):
        CircuitBreaker(open_for=-1)


def test_breaker_is_thread_safe() -> None:
    br = CircuitBreaker(failures=10_000, open_for=60)

    def hammer() -> None:
        for _ in range(1000):
            br.failure()

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert br.consecutive_failures == 8000 and not br.is_open
    for _ in range(2000):
        br.failure()
    assert br.is_open

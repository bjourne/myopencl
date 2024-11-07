import pytest

def pytest_addoption(parser):
    print(parser)
    parser.addoption(
        "--platform-index",
        action="store",
        default=0,
        help="Platform index for tests"
    )

@pytest.fixture
def platform_index(request):
    return int(request.config.getoption("--platform-index"))

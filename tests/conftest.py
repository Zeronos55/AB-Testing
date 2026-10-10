def pytest_addoption(parser):
    parser.addoption("--reps", type=int, default=10_000, help="calibration replications")

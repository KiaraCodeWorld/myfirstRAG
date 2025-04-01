## sample1 

config.json
--------

{
  "database": {
    "host": "localhost",
    "port": 5432,
    "dbname": "your_database",
    "user": "your_username",
    "password": "your_password"
  },
  "tests": [
    {
      "name": "Test 1",
      "sql": "SELECT COUNT(*) FROM your_table;",
      "expected": 100
    },
    {
      "name": "Test 2",
      "sql": "SELECT SUM(amount) FROM transactions WHERE status = 'complete';",
      "expected": 5000
    }
  ]
}

============

code1.py

import pytest
import json
import psycopg2

def read_config():
    with open('config.json', 'r') as f:
        return json.load(f)

@pytest.fixture(scope='session')
def db_connection():
    """Establish a database connection that lasts for the entire test session."""
    config = read_config()
    db_config = config['database']
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config.get('port', 5432),
        dbname=db_config['dbname'],
        user=db_config['user'],
        password=db_config['password']
    )
    yield conn
    conn.close()

def get_tests():
    """Retrieve test cases from the configuration file."""
    config = read_config()
    return config['tests']

@pytest.mark.parametrize("test_case", get_tests())
def test_sql(db_connection, test_case, request):
    sql = test_case['sql']
    expected = test_case['expected']
    test_name = test_case.get('name', 'Unnamed Test')

    cursor = db_connection.cursor()
    cursor.execute(sql)
    result = cursor.fetchone()[0]  # Assuming the query returns a single value
    cursor.close()

    # Attach extra info to the test report
    request.node._sql_executed = sql
    request.node._expected_value = expected
    request.node._actual_value = result

    # Assertion
    assert result == expected, f"Test '{test_name}' failed: Expected {expected}, got {result}"

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    config._metadata = {}

def pytest_html_results_table_header(cells):
    """Add custom columns to the HTML report header."""
    cells.insert(1, pytest_html.extras.html('<th>SQL Executed</th>'))
    cells.insert(2, pytest_html.extras.html('<th>Expected Value</th>'))
    cells.insert(3, pytest_html.extras.html('<th>Actual Value</th>'))

def pytest_html_results_table_row(report, cells):
    """Add custom data to each row in the HTML report."""
    cells.insert(1, pytest_html.extras.html(f'<td>{getattr(report, "_sql_executed", "")}</td>'))
    cells.insert(2, pytest_html.extras.html(f'<td>{getattr(report, "_expected_value", "")}</td>'))
    cells.insert(3, pytest_html.extras.html(f'<td>{getattr(report, "_actual_value", "")}</td>'))

def pytest_runtest_makereport(item, call):
    """Attach extra information to the report object."""
    outcome = yield
    report = outcome.get_result()
    for attr in ['_sql_executed', '_expected_value', '_actual_value']:
        setattr(report, attr, getattr(item, attr, ''))



====================== version2

v1.py

# conftest.py
import pytest
import psycopg2
import json

@pytest.fixture(scope="session")
def db_config():
    with open("config.json") as f:
        config = json.load(f)
    return config

@pytest.fixture(scope="session")
def db_connection(db_config):
    conn = psycopg2.connect(
        host=db_config["database"]["host"],
        port=db_config["database"]["port"],
        dbname=db_config["database"]["dbname"],
        user=db_config["database"]["user"],
        password=db_config["database"]["password"]
    )
    yield conn
    conn.close()

@pytest.fixture(scope="function")
def db_cursor(db_connection):
    cursor = db_connection.cursor()
    yield cursor
    cursor.close()


---
v2.py

# conftest.py
import pytest
import psycopg2
import json

@pytest.fixture(scope="session")
def db_config():
    with open("config.json") as f:
        config = json.load(f)
    return config

@pytest.fixture(scope="session")
def db_connection(db_config):
    conn = psycopg2.connect(
        host=db_config["database"]["host"],
        port=db_config["database"]["port"],
        dbname=db_config["database"]["dbname"],
        user=db_config["database"]["user"],
        password=db_config["database"]["password"]
    )
    yield conn
    conn.close()

@pytest.fixture(scope="function")
def db_cursor(db_connection):
    cursor = db_connection.cursor()
    yield cursor
    cursor.close()



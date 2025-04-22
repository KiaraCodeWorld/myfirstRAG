o1 preview : 

import json
import psycopg2
import psycopg2.extras
from jinja2 import Template
import datetime

# Database connection parameters
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'your_database'
DB_USER = 'your_username'
DB_PASSWORD = 'your_password'

# Path to the JSON file containing test cases and metadata
TEST_CASES_FILE = 'test_cases.json'

def load_test_cases_and_metadata(filename):
    """Load test cases and metadata from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    metadata = data.get('metadata', {})
    test_cases = data.get('test_cases', [])
    return test_cases, metadata

def execute_sql(sql, connection):
    """Execute an SQL query and return the first result."""
    with connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        cursor.execute(sql)
        result = cursor.fetchone()
        if result:
            # Assuming the result is a single value
            return list(result.values())[0]
        else:
            return None

def compare_results(actual, expected, acceptable_range):
    """Compare actual and expected results within an acceptable range."""
    if actual is None:
        return 'FAIL', actual
    try:
        actual = float(actual)
        expected = float(expected)
        acceptable_range = float(acceptable_range)
    except ValueError:
        # Non-numeric comparison
        if actual == expected:
            return 'PASS', actual
        else:
            return 'FAIL', actual
    lower_bound = expected - acceptable_range
    upper_bound = expected + acceptable_range
    if lower_bound <= actual <= upper_bound:
        return 'PASS', actual
    else:
        return 'FAIL', actual

def generate_html_report(test_results, metadata, output_file='test_report.html'):
    """Generate an HTML report from test results and metadata."""
    date_of_generation = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    template_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ metadata.report_header or 'ETL Test Report' }}</title>
        <style>
            body {font-family: Arial, sans-serif;}
            .header {margin-bottom: 20px;}
            .metadata {font-size: 14px; margin-bottom: 20px;}
            table {border-collapse: collapse; width: 100%;}
            th, td {border: 1px solid #ddd; padding: 8px;}
            tr:nth-child(even){background-color: #f9f9f9;}
            th {background-color: #4CAF50; color: white; text-align: left;}
            .PASS {color: green; font-weight: bold;}
            .FAIL {color: red; font-weight: bold;}
            pre {white-space: pre-wrap;}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{{ metadata.report_header or 'ETL Test Report' }}</h1>
            <div class="metadata">
                <p><strong>Date of Generation:</strong> {{ date_of_generation }}</p>
                {% if metadata.environment %}
                <p><strong>Environment:</strong> {{ metadata.environment }}</p>
                {% endif %}
                {% for key, value in metadata.items() %}
                    {% if key not in ['report_header', 'environment'] %}
                    <p><strong>{{ key | capitalize }}:</strong> {{ value }}</p>
                    {% endif %}
                {% endfor %}
            </div>
        </div>
        <table>
            <tr>
                <th>Test Case ID</th>
                <th>Test Case Name</th>
                <th>Description</th>
                <th>SQL Query</th>
                <th>Expected Output</th>
                <th>Actual Output</th>
                <th>Result</th>
            </tr>
            {% for result in test_results %}
            <tr>
                <td>{{ result['test_case_id'] }}</td>
                <td>{{ result['test_case_name'] }}</td>
                <td>{{ result['description'] }}</td>
                <td><pre>{{ result['sql'] }}</pre></td>
                <td>{{ result['expected_output'] }}</td>
                <td>{{ result['actual_output'] }}</td>
                <td class="{{ result['status'] }}">{{ result['status'] }}</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """
    template = Template(template_html)
    html_content = template.render(
        test_results=test_results,
        metadata=metadata,
        date_of_generation=date_of_generation
    )
    with open(output_file, 'w') as f:
        f.write(html_content)
    print(f"Report generated: {output_file}")

def main():
    # Load test cases and metadata from JSON file
    test_cases, metadata = load_test_cases_and_metadata(TEST_CASES_FILE)

    # Connect to the PostgreSQL database
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("Connected to the database.")
    except Exception as e:
        print("Database connection failed.")
        print(e)
        return

    test_results = []

    # Execute each test case
    for test_case in test_cases:
        print(f"Running Test Case ID: {test_case['test_case_id']}")
        sql = test_case['sql']
        expected_output = test_case['expected_output']
        acceptable_range = test_case.get('acceptable_range', 0)
        actual_output = execute_sql(sql, connection)
        status, actual = compare_results(actual_output, expected_output, acceptable_range)

        test_results.append({
            'test_case_id': test_case['test_case_id'],
            'test_case_name': test_case['test_case_name'],
            'description': test_case['description'],
            'sql': sql,
            'expected_output': expected_output,
            'actual_output': actual,
            'status': status
        })

    # Close the database connection
    connection.close()

    # Generate the HTML report
    generate_html_report(test_results, metadata)

if __name__ == '__main__':
    main()
--------

json : 

{
  "metadata": {
    "report_header": "Monthly ETL Test Report",
    "environment": "Production",
    "prepared_by": "Data Engineering Team",
    "project": "Sales Data Pipeline"
  },
  "test_cases": [
    {
      "test_case_id": "TC001",
      "test_case_name": "Test Total Sales",
      "description": "Verify that total sales are correct for the current month.",
      "sql": "SELECT SUM(amount) FROM sales WHERE date >= '2025-04-01'",
      "expected_output": 1000000,
      "acceptable_range": 5000
    },
    {
      "test_case_id": "TC002",
      "test_case_name": "Test Customer Count",
      "description": "Check that the number of active customers is as expected.",
      "sql": "SELECT COUNT(*) FROM customers WHERE status = 'active'",
      "expected_output": 5000,
      "acceptable_range": 100
    },
    {
      "test_case_id": "TC003",
      "test_case_name": "Test Latest Transaction Date",
      "description": "Ensure the latest transaction date is today or within acceptable range.",
      "sql": "SELECT MAX(date) FROM transactions",
      "expected_output": "2025-04-22",
      "acceptable_range": 0
    }
  ]
}





    ===================================




import json
import psycopg2
import psycopg2.extras
from jinja2 import Template

# Database connection parameters
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'your_database'
DB_USER = 'your_username'
DB_PASSWORD = 'your_password'

# Path to the JSON file containing test cases
TEST_CASES_FILE = 'test_cases.json'

def load_test_cases(filename):
    """Load test cases from a JSON file."""
    with open(filename, 'r') as f:
        test_cases = json.load(f)
    return test_cases

def execute_sql(sql, connection):
    """Execute an SQL query and return the first result."""
    with connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        cursor.execute(sql)
        result = cursor.fetchone()
        if result:
            # Assuming the result is a single value
            return list(result.values())[0]
        else:
            return None

def compare_results(actual, expected, acceptable_range):
    """Compare actual and expected results within an acceptable range."""
    if actual is None:
        return 'FAIL', actual
    try:
        actual = float(actual)
        expected = float(expected)
        acceptable_range = float(acceptable_range)
    except ValueError:
        # Non-numeric comparison
        if actual == expected:
            return 'PASS', actual
        else:
            return 'FAIL', actual
    lower_bound = expected - acceptable_range
    upper_bound = expected + acceptable_range
    if lower_bound <= actual <= upper_bound:
        return 'PASS', actual
    else:
        return 'FAIL', actual

def generate_html_report(test_results, output_file='test_report.html'):
    """Generate an HTML report from test results."""
    template_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ETL Test Report</title>
        <style>
            table {border-collapse: collapse; width: 100%;}
            th, td {border: 1px solid #ddd; padding: 8px;}
            tr:nth-child(even){background-color: #f2f2f2;}
            th {background-color: #4CAF50; color: white;}
            .PASS {color: green; font-weight: bold;}
            .FAIL {color: red; font-weight: bold;}
            pre {white-space: pre-wrap;}
        </style>
    </head>
    <body>
        <h1>ETL Test Report</h1>
        <table>
            <tr>
                <th>Test Case ID</th>
                <th>Test Case Name</th>
                <th>Description</th>
                <th>SQL Query</th>
                <th>Expected Output</th>
                <th>Actual Output</th>
                <th>Result</th>
            </tr>
            {% for result in test_results %}
            <tr>
                <td>{{ result['test_case_id'] }}</td>
                <td>{{ result['test_case_name'] }}</td>
                <td>{{ result['description'] }}</td>
                <td><pre>{{ result['sql'] }}</pre></td>
                <td>{{ result['expected_output'] }}</td>
                <td>{{ result['actual_output'] }}</td>
                <td class="{{ result['status'] }}">{{ result['status'] }}</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """
    template = Template(template_html)
    html_content = template.render(test_results=test_results)
    with open(output_file, 'w') as f:
        f.write(html_content)
    print(f"Report generated: {output_file}")

def main():
    # Load test cases from JSON file
    test_cases = load_test_cases(TEST_CASES_FILE)

    # Connect to the PostgreSQL database
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("Connected to the database.")
    except Exception as e:
        print("Database connection failed.")
        print(e)
        return

    test_results = []

    # Execute each test case
    for test_case in test_cases:
        print(f"Running Test Case ID: {test_case['test_case_id']}")
        sql = test_case['sql']
        expected_output = test_case['expected_output']
        acceptable_range = test_case.get('acceptable_range', 0)
        actual_output = execute_sql(sql, connection)
        status, actual = compare_results(actual_output, expected_output, acceptable_range)

        test_results.append({
            'test_case_id': test_case['test_case_id'],
            'test_case_name': test_case['test_case_name'],
            'description': test_case['description'],
            'sql': sql,
            'expected_output': expected_output,
            'actual_output': actual,
            'status': status
        })

    # Close the database connection
    connection.close()

    # Generate the HTML report
    generate_html_report(test_results)

if __name__ == '__main__':
    main()
-------
test_cases.json
[
  {
    "test_case_id": "TC001",
    "test_case_name": "Test Total Sales",
    "description": "Verify that total sales are correct.",
    "sql": "SELECT SUM(amount) FROM sales WHERE date >= '2025-01-01'",
    "expected_output": 1000000,
    "acceptable_range": 5000
  },
  {
    "test_case_id": "TC002",
    "test_case_name": "Test Customer Count",
    "description": "Check that the number of customers is as expected.",
    "sql": "SELECT COUNT(*) FROM customers",
    "expected_output": 5000,
    "acceptable_range": 100
  }
]


===== Claude

import json
import os
import datetime
import psycopg2
import psycopg2.extras
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from typing import Dict, List, Union, Any, Tuple

class ETLTestAutomation:
    def __init__(self, db_config: Dict[str, str], test_cases_file: str):
        """
        Initialize the ETL test automation class.
        
        Args:
            db_config: Dictionary containing database connection parameters
            test_cases_file: Path to the JSON file containing test cases
        """
        self.db_config = db_config
        self.test_cases_file = test_cases_file
        self.test_results = []
        self.connection = None
        self.cursor = None
        
    def connect_to_db(self) -> None:
        """
        Establish connection to the PostgreSQL database.
        """
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            print("Successfully connected to the database.")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to the database: {e}")
    
    def disconnect_from_db(self) -> None:
        """
        Close the database connection.
        """
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("Database connection closed.")
            
    def load_test_cases(self) -> List[Dict[str, Any]]:
        """
        Load test cases from the specified JSON file.
        
        Returns:
            List of test case dictionaries
        """
        try:
            with open(self.test_cases_file, 'r') as file:
                return json.load(file)
        except Exception as e:
            raise ValueError(f"Failed to load test cases from {self.test_cases_file}: {e}")
    
    def execute_sql_query(self, sql: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Execute a SQL query and return the results.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Tuple of (results as list of dictionaries, column names)
        """
        try:
            self.cursor.execute(sql)
            columns = [desc[0] for desc in self.cursor.description]
            results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            return results, columns
        except Exception as e:
            raise RuntimeError(f"Failed to execute SQL query: {e}\nQuery: {sql}")
    
    def compare_results(self, actual_results: List[Dict[str, Any]], 
                        expected_output: Any, 
                        comparison_type: str, 
                        acceptable_range: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Compare actual results with expected output based on comparison type.
        
        Args:
            actual_results: Results returned from the SQL query
            expected_output: Expected output to compare against
            comparison_type: Type of comparison (exact, count, range, etc.)
            acceptable_range: Range parameters for range comparison
            
        Returns:
            Tuple of (success status, failure message if any)
        """
        if comparison_type == "exact":
            # Compare exact values
            if actual_results == expected_output:
                return True, ""
            return False, f"Results do not match. Expected: {expected_output}, Actual: {actual_results}"
            
        elif comparison_type == "count":
            # Compare row count
            actual_count = len(actual_results)
            if actual_count == expected_output:
                return True, ""
            return False, f"Row count mismatch. Expected: {expected_output}, Actual: {actual_count}"
            
        elif comparison_type == "range":
            # Compare if result is within acceptable range
            if not acceptable_range:
                return False, "No acceptable range specified for range comparison"
                
            column = acceptable_range.get("column")
            min_val = acceptable_range.get("min")
            max_val = acceptable_range.get("max")
            
            if not column or min_val is None or max_val is None:
                return False, "Invalid range specification"
                
            # Check if there's at least one row (result set shouldn't be empty)
            if not actual_results:
                return False, "No results returned from query"
                
            # For simplicity, we'll check the first row for range comparison
            actual_value = actual_results[0].get(column)
            
            if actual_value is None:
                return False, f"Column {column} not found in results"
                
            if min_val <= actual_value <= max_val:
                return True, ""
            return False, f"Value out of range. Expected between {min_val} and {max_val}, Actual: {actual_value}"
            
        elif comparison_type == "not_empty":
            # Check if result set is not empty
            if actual_results:
                return True, ""
            return False, "Query returned no results"
            
        else:
            return False, f"Unsupported comparison type: {comparison_type}"
    
    def run_test_cases(self) -> None:
        """
        Run all test cases and store results.
        """
        test_cases = self.load_test_cases()
        
        for test_case in test_cases:
            test_id = test_case.get("id", "unknown")
            test_name = test_case.get("name", "Unnamed Test")
            description = test_case.get("description", "")
            sql = test_case.get("sql", "")
            comparison_type = test_case.get("comparison_type", "exact")
            expected_output = test_case.get("expected_output")
            acceptable_range = test_case.get("acceptable_range")
            
            print(f"Running test case {test_id}: {test_name}")
            
            result = {
                "id": test_id,
                "name": test_name,
                "description": description,
                "sql": sql,
                "comparison_type": comparison_type,
                "expected_output": expected_output,
                "status": "FAIL",
                "message": "",
                "actual_output": None,
                "execution_time": None
            }
            
            start_time = datetime.datetime.now()
            
            try:
                # Execute the SQL query
                actual_results, columns = self.execute_sql_query(sql)
                
                # Compare results
                success, message = self.compare_results(
                    actual_results, expected_output, comparison_type, acceptable_range
                )
                
                # Update result with actual output and status
                result["actual_output"] = actual_results
                result["status"] = "PASS" if success else "FAIL"
                result["message"] = message
                
            except Exception as e:
                result["status"] = "ERROR"
                result["message"] = str(e)
            
            end_time = datetime.datetime.now()
            result["execution_time"] = (end_time - start_time).total_seconds()
            
            self.test_results.append(result)
    
    def generate_html_report(self, output_file: str = "etl_test_report.html") -> None:
        """
        Generate HTML report for test results.
        
        Args:
            output_file: Path to the output HTML file
        """
        # Create the Jinja2 environment
        env = Environment(loader=FileSystemLoader('.'))
        
        # Create the HTML template if it doesn't exist
        template_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ETL Test Automation Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                }
                h1 {
                    color: #333;
                    border-bottom: 2px solid #ddd;
                    padding-bottom: 10px;
                }
                .summary {
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                }
                .test-case {
                    margin-bottom: 30px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                }
                .test-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }
                .test-id {
                    font-weight: bold;
                    margin-right: 10px;
                }
                .test-name {
                    font-size: 18px;
                    flex-grow: 1;
                }
                .status {
                    padding: 5px 10px;
                    border-radius: 3px;
                    font-weight: bold;
                }
                .status-PASS {
                    background-color: #d4edda;
                    color: #155724;
                }
                .status-FAIL {
                    background-color: #f8d7da;
                    color: #721c24;
                }
                .status-ERROR {
                    background-color: #f8d7da;
                    color: #721c24;
                }
                .sql-box {
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 3px;
                    font-family: monospace;
                    white-space: pre-wrap;
                    margin: 10px 0;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                .message {
                    color: #721c24;
                    margin: 10px 0;
                }
                .execution-time {
                    color: #666;
                    font-size: 0.9em;
                    margin-top: 10px;
                }
            </style>
        </head>
        <body>
            <h1>ETL Test Automation Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {{ test_results|length }}</p>
                <p>Passed: {{ test_results|selectattr('status', 'equalto', 'PASS')|list|length }}</p>
                <p>Failed: {{ test_results|selectattr('status', 'equalto', 'FAIL')|list|length }}</p>
                <p>Errors: {{ test_results|selectattr('status', 'equalto', 'ERROR')|list|length }}</p>
                <p>Generated on: {{ generation_time }}</p>
            </div>
            
            {% for result in test_results %}
            <div class="test-case">
                <div class="test-header">
                    <span class="test-id">Test ID: {{ result.id }}</span>
                    <span class="test-name">{{ result.name }}</span>
                    <span class="status status-{{ result.status }}">{{ result.status }}</span>
                </div>
                
                <p><strong>Description:</strong> {{ result.description }}</p>
                
                <div>
                    <strong>SQL Query:</strong>
                    <div class="sql-box">{{ result.sql }}</div>
                </div>
                
                <div>
                    <strong>Comparison Type:</strong> {{ result.comparison_type }}
                </div>
                
                <div>
                    <strong>Expected Output:</strong>
                    <pre>{{ result.expected_output }}</pre>
                </div>
                
                {% if result.message %}
                <div class="message">
                    <strong>Message:</strong> {{ result.message }}
                </div>
                {% endif %}
                
                {% if result.actual_output %}
                <div>
                    <strong>Actual Output:</strong>
                    <pre>{{ result.actual_output }}</pre>
                </div>
                {% endif %}
                
                <div class="execution-time">
                    Execution Time: {{ "%.3f"|format(result.execution_time) }} seconds
                </div>
            </div>
            {% endfor %}
        </body>
        </html>
        """
        
        # Write the template to a file
        template_file = "etl_test_report_template.html"
        with open(template_file, "w") as f:
            f.write(template_content)
        
        # Load the template
        template = env.get_template(template_file)
        
        # Render the template with data
        html_content = template.render(
            test_results=self.test_results,
            generation_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Write the rendered HTML to a file
        with open(output_file, "w") as f:
            f.write(html_content)
            
        print(f"Report generated: {output_file}")
        
        # Clean up the template file
        os.remove(template_file)

def main():
    # Database configuration
    db_config = {
        "host": "localhost",
        "database": "your_database",
        "user": "your_username",
        "password": "your_password",
        "port": 5432
    }
    
    # Path to test cases JSON file
    test_cases_file = "etl_test_cases.json"
    
    # Create an instance of ETLTestAutomation
    etl_tester = ETLTestAutomation(db_config, test_cases_file)
    
    try:
        # Connect to the database
        etl_tester.connect_to_db()
        
        # Run the test cases
        etl_tester.run_test_cases()
        
        # Generate HTML report
        etl_tester.generate_html_report()
    finally:
        # Disconnect from the database
        etl_tester.disconnect_from_db()

if __name__ == "__main__":
    main()

----  test_cases.json
{
  "test_cases": [
    {
      "id": "TC001",
      "name": "Check customer count",
      "description": "Verify that the total number of customers is as expected",
      "sql": "SELECT COUNT(*) as customer_count FROM customers",
      "comparison_type": "exact",
      "expected_output": [{"customer_count": 1000}]
    },
    {
      "id": "TC002",
      "name": "Verify no duplicate order IDs",
      "description": "There should be no duplicate order IDs in the orders table",
      "sql": "SELECT order_id, COUNT(*) as count FROM orders GROUP BY order_id HAVING COUNT(*) > 1",
      "comparison_type": "count",
      "expected_output": 0
    },
    {
      "id": "TC003",
      "name": "Check total sales amount",
      "description": "Verify that the total sales amount is within acceptable range",
      "sql": "SELECT SUM(amount) as total_sales FROM sales WHERE date > CURRENT_DATE - INTERVAL '30 days'",
      "comparison_type": "range",
      "expected_output": null,
      "acceptable_range": {
        "column": "total_sales",
        "min": 50000,
        "max": 100000
      }
    },
    {
      "id": "TC004",
      "name": "Verify ETL transformation of customer names",
      "description": "Customer names should be properly transformed to uppercase",
      "sql": "SELECT customer_id, customer_name FROM customers WHERE customer_name != UPPER(customer_name) LIMIT 5",
      "comparison_type": "not_empty",
      "expected_output": []
    },
    {
      "id": "TC005",
      "name": "Check date consistency",
      "description": "All orders should have timestamps later than customer creation dates",
      "sql": "SELECT o.order_id FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE o.order_date < c.creation_date",
      "comparison_type": "count",
      "expected_output": 0
    }
  ]
}

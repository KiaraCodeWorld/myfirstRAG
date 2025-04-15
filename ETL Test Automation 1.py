
=========== Deepseek 
# test_etl.py
import os
import json
import pytest
import psycopg2
from datetime import datetime
from pytest_html import extras

# Configuration - Set these via environment variables or config file
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_DATABASE", "testdb"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "port": os.getenv("DB_PORT", 5432)
}

def load_test_cases(json_file):
    with open(json_file) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    yield conn
    conn.close()

def execute_sql_query(conn, sql):
    with conn.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchone()[0]
    return result

def compare_results(expected, actual, acceptable_range):
    try:
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if '%' in acceptable_range:
                percent = float(acceptable_range.strip('%')) / 100
                lower = expected * (1 - percent)
                upper = expected * (1 + percent)
            else:
                delta = float(acceptable_range)
                lower = expected - delta
                upper = expected + delta
            
            return lower <= actual <= upper
        else:
            return str(expected) == str(actual)
    except:
        return False

def generate_report_data(test_cases):
    return {
        "application_name": test_cases["application_name"],
        "description": test_cases["description"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_cases": test_cases["test_cases"]
    }

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    extras = getattr(report, "extras", [])
    
    if "report_data" in item.funcargs:
        report_data = item.funcargs["report_data"]
        html = f"""
        <h2>{report_data['application_name']}</h2>
        <p><strong>Description:</strong> {report_data['description']}</p>
        <p><strong>Test Timestamp:</strong> {report_data['timestamp']}</p>
        <table border="1">
            <tr>
                <th>Test Case</th>
                <th>SQL Query</th>
                <th>Expected Output</th>
                <th>Actual Output</th>
                <th>Acceptable Range</th>
                <th>Status</th>
            </tr>
            {"".join(f'''
            <tr>
                <td>{tc['name']}</td>
                <td><pre>{tc['sql']}</pre></td>
                <td>{tc['expected_output']}</td>
                <td>{tc['actual_output']}</td>
                <td>{tc['acceptable_range']}</td>
                <td style="color: {'green' if tc['status'] == 'PASS' else 'red'}">{tc['status']}</td>
            </tr>
            ''' for tc in report_data['test_cases'])}
        </table>
        """
        extras.append(extras.html(html))
    report.extras = extras

def pytest_generate_tests(metafunc):
    if "test_case" in metafunc.fixturenames:
        test_cases = load_test_cases("test_cases.json")
        metafunc.parametrize("test_case", test_cases["test_cases"])

def test_etl_scenario(db_connection, test_case, request):
    actual = execute_sql_query(db_connection, test_case["sql"])
    expected = test_case["expected_output"]
    acceptable_range = test_case["acceptable_range"]
    
    # Convert to float if numeric comparison
    try:
        expected = float(expected)
        actual = float(actual)
    except:
        pass
    
    result = compare_results(expected, actual, acceptable_range)
    
    # Store results for reporting
    test_case["actual_output"] = actual
    test_case["status"] = "PASS" if result else "FAIL"
    
    # Attach report data to request
    if not hasattr(request.node, "report_data"):
        full_report = generate_report_data(load_test_cases("test_cases.json"))
        request.node.report_data = full_report
    
    assert result, f"Expected {expected}, Actual {actual} (Range: {acceptable_range})"

#Example test_cases.json:


{
    "application_name": "Sales ETL Application",
    "description": "Test cases for sales data ETL process",
    "test_cases": [
        {
            "name": "Total Sales Amount Validation",
            "sql": "SELECT SUM(sales_amount) FROM sales_data",
            "expected_output": 1000000,
            "acceptable_range": "5%"
        },
        {
            "name": "Customer Count Validation",
            "sql": "SELECT COUNT(DISTINCT customer_id) FROM sales_data",
            "expected_output": 5000,
            "acceptable_range": 100
        }
    ]
}

===================== Claude : 

import os
import json
import pytest
import pandas as pd
import psycopg2
import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Union, Tuple
import configparser
import traceback

# For HTML reporting
from py.xml import html
from pytest_html import hooks, extras


class PostgresETLTester:
    """Class to handle ETL testing with PostgreSQL database"""
    
    def __init__(self, config_path: str = "config.ini"):
        """Initialize the ETL Tester with database connection details from config"""
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        # Initialize database connection
        self.conn = self._get_db_connection()
        
        # Load test cases
        self.test_cases_path = self.config.get('Paths', 'test_cases_path', fallback='test_cases')

    def _get_db_connection(self):
        """Create PostgreSQL database connection based on configuration"""
        try:
            conn = psycopg2.connect(
                host=self.config.get('Database', 'host', fallback='localhost'),
                port=self.config.get('Database', 'port', fallback='5432'),
                database=self.config.get('Database', 'database'),
                user=self.config.get('Database', 'user'),
                password=self.config.get('Database', 'password')
            )
            return conn
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL database: {str(e)}")

    def load_test_cases(self, test_file: str) -> List[Dict[str, Any]]:
        """Load test cases from a JSON file"""
        file_path = os.path.join(self.test_cases_path, test_file)
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            return pd.read_sql_query(sql, self.conn)
        except Exception as e:
            raise RuntimeError(f"SQL execution error: {str(e)}\nQuery: {sql}")
    
    def compare_results(self, actual: pd.DataFrame, expected: Dict[str, Any], 
                        acceptable_range: Dict[str, float] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Compare actual results with expected results, considering acceptable ranges
        
        Args:
            actual: DataFrame with actual query results
            expected: Dictionary with expected results 
            acceptable_range: Dictionary with column name as key and acceptable percentage range as value
        
        Returns:
            tuple: (is_success, details)
                - is_success: True if the test passes, False otherwise
                - details: Dictionary with comparison details
        """
        result = {"passed": True, "details": [], "comparison": []}
        
        # Convert expected data to DataFrame for easier comparison
        expected_df = pd.DataFrame(expected)
        
        # Check if column counts match
        if len(actual.columns) != len(expected_df.columns):
            result["passed"] = False
            result["details"].append(f"Column count mismatch: actual={len(actual.columns)}, expected={len(expected_df.columns)}")
            
            # List the different columns
            actual_cols = set(actual.columns)
            expected_cols = set(expected_df.columns)
            missing_cols = expected_cols - actual_cols
            extra_cols = actual_cols - expected_cols
            
            if missing_cols:
                result["details"].append(f"Missing columns in actual: {missing_cols}")
            if extra_cols:
                result["details"].append(f"Extra columns in actual: {extra_cols}")
        
        # Check if row counts match
        if len(actual) != len(expected_df):
            result["passed"] = False
            result["details"].append(f"Row count mismatch: actual={len(actual)}, expected={len(expected_df)}")
        
        # Initialize comparison data structure to store the cell-by-cell comparison
        for col in expected_df.columns:
            if col in actual.columns:
                # Compare column values with acceptable ranges
                for i, expected_val in enumerate(expected_df[col]):
                    if i >= len(actual):
                        break
                    
                    actual_val = actual[col].iloc[i]
                    
                    # Prepare the comparison record
                    comparison_item = {
                        "row": i,
                        "column": col,
                        "expected": expected_val,
                        "actual": actual_val,
                        "acceptable_range": None,
                        "within_range": True
                    }
                    
                    # Handle numeric comparisons with acceptable range
                    if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
                        range_pct = 0.0  # Default no range
                        
                        # Get acceptable range for this column if specified
                        if acceptable_range and col in acceptable_range:
                            range_pct = acceptable_range[col]
                        
                        min_val = expected_val * (1 - range_pct)
                        max_val = expected_val * (1 + range_pct)
                        
                        comparison_item["acceptable_range"] = {
                            "percentage": range_pct * 100,
                            "min": min_val,
                            "max": max_val
                        }
                        
                        if not (min_val <= actual_val <= max_val):
                            result["passed"] = False
                            comparison_item["within_range"] = False
                            result["details"].append(
                                f"Value mismatch at row {i}, column {col}: "
                                f"actual={actual_val}, expected={expected_val}, "
                                f"acceptable range=[{min_val:.2f}, {max_val:.2f}]"
                            )
                    
                    # Handle non-numeric comparisons (exact match)
                    elif actual_val != expected_val:
                        result["passed"] = False
                        comparison_item["within_range"] = False
                        result["details"].append(
                            f"Value mismatch at row {i}, column {col}: "
                            f"actual={actual_val}, expected={expected_val}"
                        )
                    
                    # Add the comparison item to the results
                    result["comparison"].append(comparison_item)
            else:
                # Column not found in actual results
                result["details"].append(f"Column '{col}' not found in actual results")
        
        return result["passed"], result

    def close_connection(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()


# Pytest fixtures and hooks
@pytest.fixture(scope="session")
def etl_tester():
    """Fixture to provide ETL tester instance"""
    tester = PostgresETLTester()
    yield tester
    tester.close_connection()


# Function to collect all JSON test files
def get_test_files():
    """Find all JSON test case files"""
    config = configparser.ConfigParser()
    config.read("config.ini")
    test_cases_path = config.get('Paths', 'test_cases_path', fallback='test_cases')
    
    test_files = []
    for file in os.listdir(test_cases_path):
        if file.endswith('.json'):
            test_files.append(file)
    return test_files


# Test parametrization to run tests for each JSON file
@pytest.mark.parametrize("test_file", get_test_files())
def test_etl_cases(etl_tester, test_file):
    """Run all test cases from the specified JSON file"""
    test_cases = etl_tester.load_test_cases(test_file)
    
    for test_case in test_cases:
        # Extract test case details
        test_id = test_case.get("id", "unknown")
        test_name = test_case.get("name", f"Test Case {test_id}")
        sql_query = test_case.get("sql", "")
        expected_output = test_case.get("expected_output", {})
        acceptable_range = test_case.get("acceptable_range", {})
        
        # Store test case details for the report
        test_case_data = {
            "id": test_id,
            "name": test_name,
            "sql": sql_query,
            "expected_output": expected_output,
            "acceptable_range": acceptable_range
        }
        
        # Execute the query
        try:
            actual_output = etl_tester.execute_query(sql_query)
            test_case_data["actual_output"] = actual_output.to_dict(orient="records")
            
            # Compare results
            passed, comparison_details = etl_tester.compare_results(
                actual_output, expected_output, acceptable_range
            )
            
            test_case_data["passed"] = passed
            test_case_data["comparison_details"] = comparison_details
            
            # Store test case data for the report
            # We use a module-level variable to pass data between hooks
            pytest.current_test_data = test_case_data
            
            # Assert so pytest registers pass/fail correctly
            assert passed, f"Test case {test_name} failed: {comparison_details['details']}"
            
        except Exception as e:
            test_case_data["actual_output"] = None
            test_case_data["passed"] = False
            test_case_data["comparison_details"] = {
                "details": [str(e)],
                "comparison": [],
                "error": traceback.format_exc()
            }
            
            # Store for report
            pytest.current_test_data = test_case_data
            
            # Re-raise the exception so pytest records it
            raise


# Custom HTML report generation hooks
def pytest_html_report_title(report):
    """Set the title for the HTML report based on the application name from config"""
    config = configparser.ConfigParser()
    config.read("config.ini")
    app_name = config.get('Application', 'name', fallback='ETL Automation Test Report')
    report.title = app_name


def pytest_configure(config):
    """Add custom metadata to the HTML report"""
    config = configparser.ConfigParser()
    config.read("config.ini")
    
    app_name = config.get('Application', 'name', fallback='ETL Automation')
    app_description = config.get('Application', 'description', fallback='ETL Testing Suite')
    
    # Add metadata to the report
    config._metadata = {
        "Application": app_name,
        "Description": app_description,
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def pytest_html_results_table_header(cells):
    """Customize the results table header in the HTML report"""
    cells.insert(2, html.th("Test ID"))
    cells.insert(3, html.th("Test Name"))
    cells.insert(4, html.th("SQL Query"))
    cells.insert(5, html.th("Status"))
    cells.pop()  # Remove the links column


def pytest_html_results_table_row(report, cells):
    """Customize the results table row in the HTML report"""
    if hasattr(report, "test_id"):
        cells.insert(2, html.td(report.test_id))
    else:
        cells.insert(2, html.td("N/A"))
        
    if hasattr(report, "test_name"):
        cells.insert(3, html.td(report.test_name))
    else:
        cells.insert(3, html.td("N/A"))
        
    if hasattr(report, "sql"):
        cells.insert(4, html.td(report.sql))
    else:
        cells.insert(4, html.td("N/A"))
        
    # Use color-coded status
    if hasattr(report, "passed"):
        if report.passed:
            cells.insert(5, html.td("PASS", style="color:green; font-weight:bold"))
        else:
            cells.insert(5, html.td("FAIL", style="color:red; font-weight:bold"))
    else:
        cells.insert(5, html.td("N/A"))
        
    cells.pop()  # Remove the links column


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Custom test report logic for ETL tests
    Extends the HTML report to include detailed SQL and comparison info
    """
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call":
        # Get the test case data stored during test execution
        test_data = getattr(pytest, "current_test_data", None)
        
        if test_data:
            # Add details to the report
            report.test_id = test_data.get("id", "unknown")
            report.test_name = test_data.get("name", "Unknown Test Case")
            report.sql = test_data.get("sql", "")
            report.passed = test_data.get("passed", False)
            
            # Add detailed test information as HTML
            comparison_details = test_data.get("comparison_details", {})
            details_list = comparison_details.get("details", [])
            comparison_list = comparison_details.get("comparison", [])
            
            # Create detailed HTML content for the report
            html_content = f"""
            <div class="etl-test-details">
                <h3>Test Details: {report.test_name}</h3>
                <div class="test-metadata">
                    <p><strong>Test ID:</strong> {report.test_id}</p>
                    <p><strong>Status:</strong> <span style="color:{'green' if report.passed else 'red'}; font-weight:bold">
                        {'PASS' if report.passed else 'FAIL'}</span></p>
                </div>
                
                <div class="sql-query">
                    <h4>SQL Query:</h4>
                    <pre>{report.sql}</pre>
                </div>
            """
            
            # Add expected and actual outputs
            expected_output = test_data.get("expected_output", {})
            actual_output = test_data.get("actual_output", {})
            
            expected_df = pd.DataFrame(expected_output)
            actual_df = pd.DataFrame(actual_output) if actual_output else pd.DataFrame()
            
            # Add acceptable ranges section
            acceptable_range = test_data.get("acceptable_range", {})
            if acceptable_range:
                html_content += """
                <div class="acceptable-ranges">
                    <h4>Acceptable Ranges:</h4>
                    <table class="table table-bordered">
                        <thead>
                            <tr>
                                <th>Column</th>
                                <th>Acceptable Range (%)</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                
                for col, range_pct in acceptable_range.items():
                    html_content += f"""
                        <tr>
                            <td>{col}</td>
                            <td>±{range_pct * 100}%</td>
                        </tr>
                    """
                
                html_content += """
                        </tbody>
                    </table>
                </div>
                """
            
            # Add Expected vs Actual tables
            if not expected_df.empty or not actual_df.empty:
                html_content += """
                <div class="data-comparison">
                    <div class="row">
                        <div class="col">
                            <h4>Expected Output:</h4>
                """
                
                if not expected_df.empty:
                    expected_html = expected_df.to_html(classes="table table-bordered table-hover", index=False)
                    html_content += expected_html
                else:
                    html_content += "<p>No expected data provided</p>"
                
                html_content += """
                        </div>
                        <div class="col">
                            <h4>Actual Output:</h4>
                """
                
                if not actual_df.empty:
                    actual_html = actual_df.to_html(classes="table table-bordered table-hover", index=False)
                    html_content += actual_html
                else:
                    html_content += "<p>No actual data retrieved</p>"
                
                html_content += """
                        </div>
                    </div>
                </div>
                """
            
            # Add comparison results
            if details_list:
                html_content += """
                <div class="comparison-results">
                    <h4>Comparison Results:</h4>
                    <ul class="list-group">
                """
                
                for detail in details_list:
                    html_content += f'<li class="list-group-item list-group-item-danger">{detail}</li>'
                
                html_content += """
                    </ul>
                </div>
                """
            
            # Add detailed cell comparison if available
            if comparison_list:
                html_content += """
                <div class="detailed-comparison">
                    <h4>Detailed Cell Comparison:</h4>
                    <table class="table table-bordered">
                        <thead>
                            <tr>
                                <th>Row</th>
                                <th>Column</th>
                                <th>Expected</th>
                                <th>Actual</th>
                                <th>Acceptable Range</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                
                for comp in comparison_list:
                    if not comp.get("within_range", True):
                        row_class = "table-danger"
                        status = '<span style="color:red; font-weight:bold">FAIL</span>'
                    else:
                        row_class = "table-success"
                        status = '<span style="color:green; font-weight:bold">PASS</span>'
                    
                    range_info = comp.get("acceptable_range")
                    if range_info:
                        range_text = f"±{range_info['percentage']}% [{range_info['min']:.2f}, {range_info['max']:.2f}]"
                    else:
                        range_text = "Exact match required"
                    
                    html_content += f"""
                        <tr class="{row_class}">
                            <td>{comp['row']}</td>
                            <td>{comp['column']}</td>
                            <td>{comp['expected']}</td>
                            <td>{comp['actual']}</td>
                            <td>{range_text}</td>
                            <td>{status}</td>
                        </tr>
                    """
                
                html_content += """
                        </tbody>
                    </table>
                </div>
                """
            
            # Close the outer div
            html_content += "</div>"
            
            # Add custom styles for the report
            css_style = """
            <style>
                .etl-test-details {
                    margin: 20px 0;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                .etl-test-details h3 {
                    margin-top: 0;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #eee;
                }
                .sql-query {
                    margin: 15px 0;
                }
                .sql-query pre {
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                }
                .data-comparison .row {
                    display: flex;
                    flex-wrap: wrap;
                }
                .data-comparison .col {
                    flex: 1;
                    padding: 0 10px;
                    min-width: 300px;
                }
                table.table {
                    width: 100%;
                    margin-bottom: 1rem;
                    color: #212529;
                    border-collapse: collapse;
                }
                table.table th, table.table td {
                    padding: 0.75rem;
                    vertical-align: top;
                    border: 1px solid #dee2e6;
                }
                table.table thead th {
                    vertical-align: bottom;
                    border-bottom: 2px solid #dee2e6;
                    background-color: #f8f9fa;
                }
                .table-bordered {
                    border: 1px solid #dee2e6;
                }
                .table-hover tbody tr:hover {
                    background-color: rgba(0, 0, 0, 0.075);
                }
                .table-success {
                    background-color: #d4edda;
                }
                .table-danger {
                    background-color: #f8d7da;
                }
                .list-group {
                    display: flex;
                    flex-direction: column;
                    padding-left: 0;
                    margin-bottom: 0;
                }
                .list-group-item {
                    position: relative;
                    display: block;
                    padding: 0.75rem 1.25rem;
                    margin-bottom: -1px;
                    background-color: #fff;
                    border: 1px solid rgba(0, 0, 0, 0.125);
                }
                .list-group-item-danger {
                    color: #721c24;
                    background-color: #f8d7da;
                }
            </style>
            """
            
            # Add the HTML content and CSS to the report
            report.extra = [extras.html(css_style + html_content)]


# Sample config.ini structure
def create_sample_config():
    """Create a sample config.ini file if it doesn't exist"""
    if not os.path.exists("config.ini"):
        config = configparser.ConfigParser()
        
        config["Application"] = {
            "name": "ETL Testing Framework",
            "description": "Automated ETL testing for PostgreSQL databases"
        }
        
        config["Database"] = {
            "host": "localhost",
            "port": "5432",
            "database": "your_database",
            "user": "your_username",
            "password": "your_password"  # In production, use environment variables instead
        }
        
        config["Paths"] = {
            "test_cases_path": "test_cases",
            "reports_path": "reports"
        }
        
        with open("config.ini", "w") as f:
            config.write(f)


# Sample test case structure
def create_sample_test_case():
    """Create a sample test case JSON file if the test_cases directory doesn't exist"""
    os.makedirs("test_cases", exist_ok=True)
    
    sample_test_case = {
        "test_cases": [
            {
                "id": "ETL001",
                "name": "Customer Count Validation",
                "sql": "SELECT count(*) as customer_count FROM customers",
                "expected_output": [
                    {"customer_count": 1000}
                ],
                "acceptable_range": {
                    "customer_count": 0.05  # 5% acceptable range
                }
            },
            {
                "id": "ETL002",
                "name": "Revenue Calculation Validation",
                "sql": "SELECT SUM(amount) as total_revenue FROM orders WHERE order_date >= '2023-01-01'",
                "expected_output": [
                    {"total_revenue": 150000.00}
                ],
                "acceptable_range": {
                    "total_revenue": 0.02  # 2% acceptable range
                }
            },
            {
                "id": "ETL003",
                "name": "Data Transformation Validation",
                "sql": "SELECT customer_id, first_name, last_name, email FROM customers WHERE customer_id < 5 ORDER BY customer_id",
                "expected_output": [
                    {"customer_id": 1, "first_name": "John", "last_name": "Doe", "email": "john.doe@example.com"},
                    {"customer_id": 2, "first_name": "Jane", "last_name": "Smith", "email": "jane.smith@example.com"},
                    {"customer_id": 3, "first_name": "Robert", "last_name": "Johnson", "email": "robert.j@example.com"},
                    {"customer_id": 4, "first_name": "Sarah", "last_name": "Williams", "email": "sarah.w@example.com"}
                ],
                "acceptable_range": {}  # Empty dict means exact match required
            }
        ]
    }
    
    with open("test_cases/sample_etl_tests.json", "w") as f:
        json.dump(sample_test_case, f, indent=4)


# Main function to demonstrate usage
if __name__ == "__main__":
    # Create sample config and test case files
    create_sample_config()
    create_sample_test_case()
    
    print("Sample configuration and test case files created.")
    print("To run the tests, execute: pytest -v test_etl.py --html=report.html")


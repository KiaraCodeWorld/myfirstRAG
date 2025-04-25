code -- O1 preview : 

import json
import psycopg2
import psycopg2.extras
from jinja2 import Template
import datetime
import pandas as pd  # Added for CSV and DataFrame handling

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

def execute_sql_fetchall(sql, connection):
    """Execute an SQL query and return all results as a list of dictionaries."""
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(sql)
        results = cursor.fetchall()
        return results

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

def compare_csv_and_db_data(csv_data, db_data, comparison_keys, acceptable_ranges):
    """Compare CSV data and database data based on comparison keys and acceptable ranges."""
    total_records = len(csv_data)
    status = 'PASS'

    # Merge data on comparison keys
    merged_data = pd.merge(
        csv_data, db_data, on=comparison_keys, how='outer', indicator=True, suffixes=('_csv', '_db')
    )

    # Find records only in CSV or only in DB
    only_in_csv = merged_data[merged_data['_merge'] == 'left_only']
    only_in_db = merged_data[merged_data['_merge'] == 'right_only']

    # Records present in both, compare columns
    in_both = merged_data[merged_data['_merge'] == 'both']

    mismatches = pd.DataFrame()
    for col in csv_data.columns:
        if col in comparison_keys:
            continue  # Skip comparison keys
        col_csv = f"{col}_csv"
        col_db = f"{col}_db"

        # Check if acceptable range is specified for the column
        range_val = acceptable_ranges.get(col, 0)

        if range_val:
            # For numeric columns with acceptable range
            try:
                in_both[col_csv] = pd.to_numeric(in_both[col_csv], errors='coerce')
                in_both[col_db] = pd.to_numeric(in_both[col_db], errors='coerce')
                in_both['diff'] = abs(in_both[col_csv] - in_both[col_db])
                mismatches_col = in_both[in_both['diff'] > range_val]
            except Exception as e:
                # Non-numeric comparison or conversion error
                mismatches_col = in_both[in_both[col_csv] != in_both[col_db]]
        else:
            # Compare columns directly for exact match
            mismatches_col = in_both[in_both[col_csv] != in_both[col_db]]

        mismatches = pd.concat([mismatches, mismatches_col])

    total_mismatches = len(only_in_csv) + len(only_in_db) + len(mismatches)

    if total_mismatches > 0:
        status = 'FAIL'

    return status, total_mismatches, total_records

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

        if 'csv_file' in test_case:
            # CSV to database comparison
            csv_file = test_case['csv_file']
            sql = test_case['sql']
            comparison_keys = test_case['comparison_keys']
            acceptable_ranges = test_case.get('column_acceptable_ranges', {})

            # Load CSV data
            try:
                csv_data = pd.read_csv(csv_file)
                print(f"Loaded CSV data from {csv_file}.")
            except Exception as e:
                print(f"Failed to load CSV file {csv_file}.")
                print(e)
                status = 'FAIL'
                test_results.append({
                    'test_case_id': test_case['test_case_id'],
                    'test_case_name': test_case['test_case_name'],
                    'description': test_case['description'],
                    'sql': sql,
                    'expected_output': 'CSV data could not be loaded.',
                    'actual_output': str(e),
                    'status': status
                })
                continue  # Move to the next test case

            # Fetch data from database
            try:
                db_data = execute_sql_fetchall(sql, connection)
                db_data = pd.DataFrame(db_data)
                print(f"Fetched data from database for SQL: {sql}")
            except Exception as e:
                print(f"Failed to execute SQL query: {sql}")
                print(e)
                status = 'FAIL'
                test_results.append({
                    'test_case_id': test_case['test_case_id'],
                    'test_case_name': test_case['test_case_name'],
                    'description': test_case['description'],
                    'sql': sql,
                    'expected_output': 'Database query failed.',
                    'actual_output': str(e),
                    'status': status
                })
                continue  # Move to the next test case

            # Compare CSV data and database data
            status, mismatches, total_records = compare_csv_and_db_data(
                csv_data, db_data, comparison_keys, acceptable_ranges
            )
            actual_output = f'Mismatches Found: {mismatches} out of {total_records} records'

            test_results.append({
                'test_case_id': test_case['test_case_id'],
                'test_case_name': test_case['test_case_name'],
                'description': test_case['description'],
                'sql': sql,
                'expected_output': 'CSV data should match database data.',
                'actual_output': actual_output,
                'status': status
            })
        else:
            # Existing test case processing for static expected_output
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

====== test_cases.json

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
    },
    {
      "test_case_id": "TC004",
      "test_case_name": "Compare CSV to Database",
      "description": "Ensure that data in the CSV matches data in the database.",
      "csv_file": "data/source_data.csv",
      "sql": "SELECT * FROM target_table",
      "comparison_keys": ["id"],
      "column_acceptable_ranges": {
        "amount": 5.0,  
        "date": 0    
      }
    }
  ]
}

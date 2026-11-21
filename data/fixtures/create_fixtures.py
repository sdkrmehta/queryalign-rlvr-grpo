"""Creates a deterministic SQLite fixture database for unit and security tests."""

import sqlite3
from pathlib import Path


def create_test_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE department (
        dept_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        budget REAL
    );
    """)

    cur.execute("""
    CREATE TABLE employee (
        emp_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        dept_id INTEGER,
        salary REAL,
        FOREIGN KEY (dept_id) REFERENCES department (dept_id)
    );
    """)

    departments = [
        (1, "Engineering", 500000.0),
        (2, "Sales", 250000.0),
        (3, "HR", 100000.0),
    ]
    cur.executemany("INSERT INTO department VALUES (?, ?, ?);", departments)

    employees = [
        (101, "Alice", 1, 95000.0),
        (102, "Bob", 1, 85000.0),
        (103, "Charlie", 2, 70000.0),
        (104, "David", 3, 60000.0),
        (105, "Eve", 2, 75000.0),
    ]
    cur.executemany("INSERT INTO employee VALUES (?, ?, ?, ?);", employees)

    conn.commit()
    conn.close()
    print(f"Test database created successfully at: {db_path.resolve()}")


if __name__ == "__main__":
    create_test_db(Path("data/fixtures/test_company.sqlite"))

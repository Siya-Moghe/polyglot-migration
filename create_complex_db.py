"""
create_complex_db.py
Builds an enterprise IT Helpdesk & HR database with deep relationships.
"""
import sqlite3, os

DB_PATH = "enterprise_system.db"

def create_db():
    if os.path.exists(DB_PATH): os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. Departments (Isolated entity - good for Mongo)
    cur.execute('''CREATE TABLE departments (
        id INTEGER PRIMARY KEY, name TEXT, cost_center TEXT, location TEXT)''')

    # 2. Employees (Core entity with FK)
    cur.execute('''CREATE TABLE employees (
        id INTEGER PRIMARY KEY, name TEXT, email TEXT, role TEXT, 
        dept_id INTEGER, FOREIGN KEY(dept_id) REFERENCES departments(id))''')

    # 3. Knowledge Base (Heavy text - perfect for Chroma/Vector)
    cur.execute('''CREATE TABLE knowledge_base (
        id INTEGER PRIMARY KEY, title TEXT, full_content TEXT, 
        author_id INTEGER, tags TEXT, 
        FOREIGN KEY(author_id) REFERENCES employees(id))''')

    # 4. Projects (Mixed text and metadata)
    cur.execute('''CREATE TABLE projects (
        id INTEGER PRIMARY KEY, project_name TEXT, objective TEXT, deadline TEXT)''')

    # 5. Employee_Projects (Many-to-Many Mapping - perfect for Neo4j/Graph)
    cur.execute('''CREATE TABLE employee_projects (
        emp_id INTEGER, proj_id INTEGER, allocation_percentage INTEGER,
        PRIMARY KEY(emp_id, proj_id),
        FOREIGN KEY(emp_id) REFERENCES employees(id),
        FOREIGN KEY(proj_id) REFERENCES projects(id))''')

    # 6. IT Tickets (Transactional records)
    cur.execute('''CREATE TABLE support_tickets (
        id INTEGER PRIMARY KEY, requester_id INTEGER, issue_summary TEXT, 
        status TEXT, priority TEXT,
        FOREIGN KEY(requester_id) REFERENCES employees(id))''')

    # --- Insert Complex Mock Data ---
    cur.executemany("INSERT INTO departments VALUES (?,?,?,?)", [
        (1, "Engineering", "CC-101", "New York"), (2, "HR", "CC-202", "London"),
        (3, "IT Support", "CC-303", "Remote")
    ])
    cur.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", [
        (101, "Alice Smith", "alice@corp.com", "Backend Dev", 1),
        (102, "Bob Jones", "bob@corp.com", "HR Manager", 2),
        (103, "Charlie Day", "charlie@corp.com", "SysAdmin", 3),
        (104, "Diana Prince", "diana@corp.com", "Frontend Dev", 1)
    ])
    cur.executemany("INSERT INTO knowledge_base VALUES (?,?,?,?,?)", [
        (1, "VPN Setup Guide", "To connect to the corporate VPN, download the Cisco AnyConnect client. Enter gateway.corp.com. Use your standard SSO credentials to authenticate. If you face DNS issues, flush your DNS cache.", 103, "IT, VPN, Network"),
        (2, "Q3 Engineering Guidelines", "All new microservices must be written in Python or Go. Use Docker for containerization. CI/CD pipelines must include 80% test coverage gates before merging to main.", 101, "Engineering, Policy")
    ])
    cur.executemany("INSERT INTO projects VALUES (?,?,?,?)", [
        (1, "Cloud Migration 2026", "Migrate all legacy on-prem SQL servers to AWS RDS instances.", "2026-12-01"),
        (2, "Employee Portal V2", "Revamp the HR benefits dashboard using Next.js and Tailwind CSS.", "2026-08-15")
    ])
    cur.executemany("INSERT INTO employee_projects VALUES (?,?,?)", [
        (101, 1, 50), (103, 1, 100), (104, 2, 80), (102, 2, 20)
    ])
    cur.executemany("INSERT INTO support_tickets VALUES (?,?,?,?,?)", [
        (1, 104, "My Docker container keeps crashing on startup", "Open", "High"),
        (2, 102, "Need access to the new payroll software", "Resolved", "Medium")
    ])

    conn.commit()
    conn.close()
    print(f"Database created: {DB_PATH}")

if __name__ == "__main__": create_db()
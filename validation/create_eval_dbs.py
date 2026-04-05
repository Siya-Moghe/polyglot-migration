"""
validation/create_eval_dbs.py

Generates 40 SQLite evaluation databases in validation/ground_tables/.
Each .db file contains one or more tables with clear routing characteristics.

Run from your project root:
    python -m validation.create_eval_dbs

Ground truth routing labels are embedded as comments and also exported to
ground_truth.json in the validation/ folder (consumed by run_eval.py).
"""

import os
import json
import sqlite3

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "ground_tables")
GT_PATH    = os.path.join(os.path.dirname(__file__), "ground_truth.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Master definition list
# Each entry:
#   db_file      : filename (goes inside ground_tables/)
#   table_name   : the single table in that db
#   expected_db  : "chroma" | "mongo" | "neo4j" | "relational"
#   confidence   : "high" | "medium"  (high = unambiguous, medium = debatable)
#   rationale    : human reasoning for the label
#   ddl          : CREATE TABLE SQL
#   rows         : list of tuples to INSERT
# ─────────────────────────────────────────────────────────────────────────────
EVAL_TABLES = [

    # ══════════════════════════════════════════════════════════════════
    # CHROMA — semantic / long-text tables (10 cases)
    # ══════════════════════════════════════════════════════════════════

    {
        "db_file": "chroma_01_knowledge_base.db",
        "table_name": "knowledge_base",
        "expected_db": "chroma",
        "confidence": "high",
        "rationale": "full_content is long-form text — canonical vector embedding use case.",
        "ddl": """
            CREATE TABLE knowledge_base (
                id          INTEGER PRIMARY KEY,
                title       TEXT,
                full_content TEXT,
                author_id   INTEGER,
                tags        TEXT,
                FOREIGN KEY (author_id) REFERENCES employees(id)
            )""",
        "rows": [
            (1, "VPN Setup Guide",
             "To connect to the corporate VPN, download the Cisco AnyConnect client. "
             "Enter gateway.corp.com as the server. Use your SSO credentials. "
             "If you face DNS issues, flush your DNS cache with ipconfig /flushdns.",
             103, "IT, VPN, Network"),
            (2, "Q3 Engineering Guidelines",
             "All new microservices must be written in Python or Go. "
             "Use Docker for containerization. CI/CD pipelines must include "
             "80% test coverage gates before merging to main branch.",
             101, "Engineering, Policy"),
        ],
    },
    {
        "db_file": "chroma_02_product_reviews.db",
        "table_name": "product_reviews",
        "expected_db": "chroma",
        "confidence": "high",
        "rationale": "review_text is long user-generated content — semantic similarity search.",
        "ddl": """
            CREATE TABLE product_reviews (
                id          INTEGER PRIMARY KEY,
                product_id  INTEGER,
                reviewer    TEXT,
                review_text TEXT,
                rating      INTEGER
            )""",
        "rows": [
            (1, 42, "Alice",
             "This laptop exceeded all my expectations. Battery life is phenomenal "
             "and the keyboard is incredibly comfortable for long coding sessions. "
             "Highly recommend for developers.", 5),
            (2, 42, "Bob",
             "Good value for money but the fan noise is quite loud under heavy load. "
             "Not ideal for quiet office environments but fine for home use.", 3),
            (3, 17, "Carol",
             "The build quality feels cheap compared to the price point. "
             "Returned after two weeks due to the trackpad being unresponsive.", 1),
        ],
    },
    {
        "db_file": "chroma_03_research_papers.db",
        "table_name": "research_papers",
        "expected_db": "chroma",
        "confidence": "high",
        "rationale": "abstract and body are scholarly long text — vector similarity retrieval.",
        "ddl": """
            CREATE TABLE research_papers (
                paper_id  INTEGER PRIMARY KEY,
                title     TEXT,
                abstract  TEXT,
                body      TEXT,
                author_id INTEGER,
                keywords  TEXT
            )""",
        "rows": [
            (1, "Transformer Architectures in NLP",
             "This paper surveys the evolution of transformer-based models from BERT to GPT-4, "
             "analyzing their impact on downstream NLP tasks including QA and summarization.",
             "Full paper body text...", 7, "NLP, transformers, BERT, GPT"),
            (2, "Efficient Graph Neural Networks",
             "We propose GraphLite, a lightweight GNN that achieves competitive accuracy "
             "on node classification benchmarks while reducing memory by 60% vs GraphSAGE.",
             "Full paper body text...", 12, "GNN, efficiency, node classification"),
        ],
    },
    {
        "db_file": "chroma_04_job_postings.db",
        "table_name": "job_postings",
        "expected_db": "chroma",
        "confidence": "high",
        "rationale": "description is a long job posting text — candidates search semantically.",
        "ddl": """
            CREATE TABLE job_postings (
                job_id      INTEGER PRIMARY KEY,
                title       TEXT,
                description TEXT,
                location    TEXT,
                salary_min  INTEGER,
                salary_max  INTEGER
            )""",
        "rows": [
            (1, "Senior Backend Engineer",
             "We are looking for a senior backend engineer with 5+ years of experience in "
             "Python, distributed systems, and cloud infrastructure. You will design and "
             "implement scalable microservices for our data platform team. Strong knowledge "
             "of Kafka, Redis, and PostgreSQL required.", "Remote", 120000, 160000),
            (2, "Product Designer",
             "Join our design team to craft intuitive user experiences for millions of users. "
             "You will work closely with product managers and engineers to define the visual "
             "language and interaction patterns across our mobile and web products.", "New York", 90000, 130000),
        ],
    },
    {
        "db_file": "chroma_05_support_articles.db",
        "table_name": "support_articles",
        "expected_db": "chroma",
        "confidence": "high",
        "rationale": "content is helpdesk article body text — keyword/semantic FAQ search.",
        "ddl": """
            CREATE TABLE support_articles (
                article_id  INTEGER PRIMARY KEY,
                category    TEXT,
                question    TEXT,
                content     TEXT,
                updated_at  TEXT
            )""",
        "rows": [
            (1, "Billing",
             "How do I update my payment method?",
             "To update your payment method, navigate to Account Settings > Billing > "
             "Payment Methods. Click 'Add New Card' and enter your card details. "
             "Your existing subscriptions will automatically use the new card.", "2024-01-15"),
            (2, "Technical",
             "Why is my export taking too long?",
             "Large exports may take several minutes depending on the dataset size. "
             "We recommend scheduling exports during off-peak hours (midnight to 6am). "
             "If your export exceeds 2 hours, please contact support with your export ID.", "2024-02-20"),
        ],
    },
    {
        "db_file": "chroma_06_blog_posts.db",
        "table_name": "blog_posts",
        "expected_db": "chroma",
        "confidence": "high",
        "rationale": "body is full blog post content — semantic search and recommendation.",
        "ddl": """
            CREATE TABLE blog_posts (
                post_id     INTEGER PRIMARY KEY,
                title       TEXT,
                body        TEXT,
                author      TEXT,
                published   TEXT,
                tags        TEXT
            )""",
        "rows": [
            (1, "Getting Started with LangGraph",
             "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
             "In this post we walk through building your first agent graph, adding conditional "
             "edges, and implementing reflection loops for self-correction.",
             "Jane Doe", "2024-03-01", "AI, LangGraph, Agents"),
            (2, "Why Polyglot Persistence Matters",
             "Modern applications rarely fit into a single database paradigm. Using the right "
             "tool for each data shape — relational for transactions, vector for search, "
             "graph for relationships — dramatically improves both performance and developer ergonomics.",
             "John Smith", "2024-04-15", "Database, Architecture"),
        ],
    },
    {
        "db_file": "chroma_07_legal_contracts.db",
        "table_name": "legal_contracts",
        "expected_db": "chroma",
        "confidence": "high",
        "rationale": "contract_text is dense legal prose — semantic clause retrieval.",
        "ddl": """
            CREATE TABLE legal_contracts (
                contract_id   INTEGER PRIMARY KEY,
                party_a       TEXT,
                party_b       TEXT,
                contract_text TEXT,
                signed_date   TEXT,
                contract_type TEXT
            )""",
        "rows": [
            (1, "Acme Corp", "BuildRight LLC",
             "This Service Agreement ('Agreement') is entered into as of January 1 2024 between "
             "Acme Corp ('Client') and BuildRight LLC ('Vendor'). Vendor agrees to provide software "
             "development services as described in Schedule A. Payment terms are Net-30 from invoice date. "
             "Either party may terminate with 30 days written notice.",
             "2024-01-01", "Service Agreement"),
        ],
    },
    {
        "db_file": "chroma_08_medical_notes.db",
        "table_name": "medical_notes",
        "expected_db": "chroma",
        "confidence": "high",
        "rationale": "clinical_note is long unstructured clinical text — semantic retrieval for diagnosis support.",
        "ddl": """
            CREATE TABLE medical_notes (
                note_id      INTEGER PRIMARY KEY,
                patient_id   INTEGER,
                doctor_id    INTEGER,
                clinical_note TEXT,
                visit_date   TEXT
            )""",
        "rows": [
            (1, 1001, 55,
             "Patient presents with persistent dry cough for 3 weeks, mild fever, and fatigue. "
             "No significant travel history. Chest X-ray shows mild bilateral infiltrates. "
             "Prescribed azithromycin 500mg for 5 days and advised rest. Follow-up in 1 week.",
             "2024-05-10"),
        ],
    },
    {
        "db_file": "chroma_09_course_descriptions.db",
        "table_name": "courses",
        "expected_db": "chroma",
        "confidence": "high",
        "rationale": "syllabus and description are long educational text — semantic course search.",
        "ddl": """
            CREATE TABLE courses (
                course_id   INTEGER PRIMARY KEY,
                title       TEXT,
                description TEXT,
                syllabus    TEXT,
                instructor  TEXT,
                credits     INTEGER
            )""",
        "rows": [
            (1, "Advanced Machine Learning",
             "This course covers modern ML techniques including deep learning, reinforcement "
             "learning, and large language models. Students will implement algorithms from scratch "
             "and apply them to real-world datasets.",
             "Week 1: Linear Algebra Review. Week 2: Neural Networks. Week 3: CNNs. "
             "Week 4: RNNs and Transformers. Week 5: RL Fundamentals. Week 6: LLMs.",
             "Dr. Sarah Chen", 4),
        ],
    },
    {
        "db_file": "chroma_10_news_articles.db",
        "table_name": "news_articles",
        "expected_db": "chroma",
        "confidence": "high",
        "rationale": "content is full news article body — topic similarity and search.",
        "ddl": """
            CREATE TABLE news_articles (
                article_id  INTEGER PRIMARY KEY,
                headline    TEXT,
                content     TEXT,
                source      TEXT,
                published   TEXT,
                category    TEXT
            )""",
        "rows": [
            (1, "Central Bank Raises Interest Rates by 25 Basis Points",
             "The Federal Reserve raised its benchmark interest rate by 25 basis points on Wednesday, "
             "bringing the federal funds rate to a 22-year high. Fed Chair Jerome Powell indicated "
             "that further hikes remain on the table depending on upcoming inflation data. "
             "Markets reacted with a sell-off in treasury bonds and a brief equity dip.",
             "Reuters", "2024-07-31", "Finance"),
        ],
    },

    # ══════════════════════════════════════════════════════════════════
    # MONGO — self-contained entity / document tables (10 cases)
    # ══════════════════════════════════════════════════════════════════

    {
        "db_file": "mongo_01_departments.db",
        "table_name": "departments",
        "expected_db": "mongo",
        "confidence": "high",
        "rationale": "Self-contained entity, no FKs, short text fields — document store.",
        "ddl": """
            CREATE TABLE departments (
                id          INTEGER PRIMARY KEY,
                name        TEXT,
                cost_center TEXT,
                location    TEXT
            )""",
        "rows": [
            (1, "Engineering", "CC-101", "New York"),
            (2, "HR", "CC-202", "London"),
            (3, "IT Support", "CC-303", "Remote"),
        ],
    },
    {
        "db_file": "mongo_02_user_profiles.db",
        "table_name": "user_profiles",
        "expected_db": "mongo",
        "confidence": "high",
        "rationale": "Flat user entity with mixed field types and no FKs — document store.",
        "ddl": """
            CREATE TABLE user_profiles (
                user_id     INTEGER PRIMARY KEY,
                username    TEXT,
                email       TEXT,
                bio         TEXT,
                avatar_url  TEXT,
                country     TEXT,
                timezone    TEXT
            )""",
        "rows": [
            (1, "alice99", "alice@example.com", "Software engineer and coffee enthusiast.",
             "https://cdn.example.com/avatars/alice.png", "USA", "America/New_York"),
            (2, "bob_dev", "bob@example.com", "Open source contributor. Python lover.",
             "https://cdn.example.com/avatars/bob.png", "UK", "Europe/London"),
        ],
    },
    {
        "db_file": "mongo_03_product_catalog.db",
        "table_name": "products",
        "expected_db": "mongo",
        "confidence": "high",
        "rationale": "Product entity with mixed text/numeric attributes and no FKs — document store.",
        "ddl": """
            CREATE TABLE products (
                product_id  INTEGER PRIMARY KEY,
                name        TEXT,
                brand       TEXT,
                category    TEXT,
                price       REAL,
                stock       INTEGER,
                sku         TEXT
            )""",
        "rows": [
            (1, "Wireless Noise-Cancelling Headphones", "SoundMax", "Electronics", 299.99, 150, "SM-WH-001"),
            (2, "Ergonomic Office Chair", "ComfortPro", "Furniture", 499.00, 42, "CP-OC-002"),
            (3, "Mechanical Keyboard TKL", "KeyTech", "Electronics", 129.99, 320, "KT-KB-003"),
        ],
    },
    {
        "db_file": "mongo_04_company_locations.db",
        "table_name": "offices",
        "expected_db": "mongo",
        "confidence": "high",
        "rationale": "Reference/config entity with no FKs and mixed short fields — document store.",
        "ddl": """
            CREATE TABLE offices (
                office_id   INTEGER PRIMARY KEY,
                city        TEXT,
                country     TEXT,
                address     TEXT,
                phone       TEXT,
                capacity    INTEGER,
                timezone    TEXT
            )""",
        "rows": [
            (1, "San Francisco", "USA", "100 Market St, Suite 500", "+1-415-555-0100", 250, "America/Los_Angeles"),
            (2, "Berlin", "Germany", "Unter den Linden 77", "+49-30-555-0200", 80, "Europe/Berlin"),
        ],
    },
    {
        "db_file": "mongo_05_config_settings.db",
        "table_name": "app_config",
        "expected_db": "mongo",
        "confidence": "high",
        "rationale": "Key-value config entity, no FKs, no relational structure — document store.",
        "ddl": """
            CREATE TABLE app_config (
                config_id   INTEGER PRIMARY KEY,
                module      TEXT,
                key         TEXT,
                value       TEXT,
                description TEXT,
                env         TEXT
            )""",
        "rows": [
            (1, "auth", "session_timeout", "3600", "Session timeout in seconds", "production"),
            (2, "email", "smtp_host", "smtp.sendgrid.net", "SMTP server hostname", "production"),
            (3, "storage", "max_file_size_mb", "50", "Maximum allowed upload size", "production"),
        ],
    },
    {
        "db_file": "mongo_06_inventory_items.db",
        "table_name": "inventory",
        "expected_db": "mongo",
        "confidence": "high",
        "rationale": "Self-contained inventory item entity with no FK references — document store.",
        "ddl": """
            CREATE TABLE inventory (
                item_id     INTEGER PRIMARY KEY,
                name        TEXT,
                category    TEXT,
                quantity    INTEGER,
                unit        TEXT,
                location    TEXT,
                supplier    TEXT
            )""",
        "rows": [
            (1, "A4 Copy Paper", "Stationery", 500, "ream", "Shelf B3", "OfficeDepot"),
            (2, "USB-C Cable 2m", "Electronics", 75, "unit", "Cabinet 4A", "TechDistrib"),
            (3, "Whiteboard Markers", "Stationery", 120, "unit", "Shelf B1", "OfficeDepot"),
        ],
    },
    {
        "db_file": "mongo_07_country_metadata.db",
        "table_name": "countries",
        "expected_db": "mongo",
        "confidence": "high",
        "rationale": "Pure lookup/reference table with no FKs — document store.",
        "ddl": """
            CREATE TABLE countries (
                country_code TEXT PRIMARY KEY,
                name         TEXT,
                continent    TEXT,
                currency     TEXT,
                calling_code TEXT,
                capital      TEXT
            )""",
        "rows": [
            ("US", "United States", "North America", "USD", "+1", "Washington D.C."),
            ("IN", "India", "Asia", "INR", "+91", "New Delhi"),
            ("DE", "Germany", "Europe", "EUR", "+49", "Berlin"),
        ],
    },
    {
        "db_file": "mongo_08_event_catalog.db",
        "table_name": "events",
        "expected_db": "mongo",
        "confidence": "high",
        "rationale": "Event entity with short descriptive fields and no FKs — document store.",
        "ddl": """
            CREATE TABLE events (
                event_id    INTEGER PRIMARY KEY,
                name        TEXT,
                venue       TEXT,
                city        TEXT,
                start_date  TEXT,
                end_date    TEXT,
                capacity    INTEGER,
                category    TEXT
            )""",
        "rows": [
            (1, "PyConf India 2025", "NIMHANS Convention Centre", "Bengaluru", "2025-09-12", "2025-09-14", 1500, "Tech"),
            (2, "Design Summit APAC", "Marina Bay Sands", "Singapore", "2025-11-03", "2025-11-04", 800, "Design"),
        ],
    },
    {
        "db_file": "mongo_09_vehicle_fleet.db",
        "table_name": "vehicles",
        "expected_db": "mongo",
        "confidence": "medium",
        "rationale": "Self-contained entity with no FKs and varied attributes — document store. Medium because fleet management often uses relational.",
        "ddl": """
            CREATE TABLE vehicles (
                vehicle_id  INTEGER PRIMARY KEY,
                make        TEXT,
                model       TEXT,
                year        INTEGER,
                vin         TEXT,
                color       TEXT,
                fuel_type   TEXT,
                mileage     INTEGER
            )""",
        "rows": [
            (1, "Toyota", "Camry", 2022, "1HGBH41JXMN109186", "Silver", "Petrol", 24500),
            (2, "Tesla", "Model 3", 2023, "5YJ3E1EA1NF123456", "White", "Electric", 8200),
        ],
    },
    {
        "db_file": "mongo_10_currency_rates.db",
        "table_name": "currency_rates",
        "expected_db": "mongo",
        "confidence": "medium",
        "rationale": "Reference document entity with no FKs. Medium because rates could also go relational for time-series querying.",
        "ddl": """
            CREATE TABLE currency_rates (
                id          INTEGER PRIMARY KEY,
                base        TEXT,
                target      TEXT,
                rate        REAL,
                date        TEXT,
                source      TEXT
            )""",
        "rows": [
            (1, "USD", "EUR", 0.9215, "2024-07-01", "ECB"),
            (2, "USD", "INR", 83.47,  "2024-07-01", "ECB"),
            (3, "USD", "GBP", 0.7891, "2024-07-01", "ECB"),
        ],
    },

    # ══════════════════════════════════════════════════════════════════
    # NEO4J — junction / relationship-heavy tables (10 cases)
    # ══════════════════════════════════════════════════════════════════

    {
        "db_file": "neo4j_01_employee_projects.db",
        "table_name": "employee_projects",
        "expected_db": "neo4j",
        "confidence": "high",
        "rationale": "Pure junction table with 2 FKs — canonical graph edge.",
        "ddl": """
            CREATE TABLE employee_projects (
                emp_id                INTEGER,
                proj_id               INTEGER,
                allocation_percentage INTEGER,
                PRIMARY KEY (emp_id, proj_id),
                FOREIGN KEY (emp_id)  REFERENCES employees(id),
                FOREIGN KEY (proj_id) REFERENCES projects(id)
            )""",
        "rows": [(101, 1, 50), (103, 1, 100), (104, 2, 80), (102, 2, 20)],
    },
    {
        "db_file": "neo4j_02_user_follows.db",
        "table_name": "user_follows",
        "expected_db": "neo4j",
        "confidence": "high",
        "rationale": "Social graph follower/followee relationship — pure graph edge.",
        "ddl": """
            CREATE TABLE user_follows (
                follower_id  INTEGER,
                followee_id  INTEGER,
                followed_at  TEXT,
                PRIMARY KEY (follower_id, followee_id),
                FOREIGN KEY (follower_id) REFERENCES users(id),
                FOREIGN KEY (followee_id) REFERENCES users(id)
            )""",
        "rows": [
            (1, 2, "2024-01-10"), (1, 3, "2024-02-05"),
            (2, 4, "2024-03-01"), (3, 1, "2024-03-15"),
        ],
    },
    {
        "db_file": "neo4j_03_product_categories.db",
        "table_name": "product_categories",
        "expected_db": "neo4j",
        "confidence": "high",
        "rationale": "Product-to-category many-to-many mapping — graph relationship.",
        "ddl": """
            CREATE TABLE product_categories (
                product_id  INTEGER,
                category_id INTEGER,
                PRIMARY KEY (product_id, category_id),
                FOREIGN KEY (product_id)  REFERENCES products(id),
                FOREIGN KEY (category_id) REFERENCES categories(id)
            )""",
        "rows": [(1, 10), (1, 12), (2, 11), (3, 10), (3, 13)],
    },
    {
        "db_file": "neo4j_04_role_permissions.db",
        "table_name": "role_permissions",
        "expected_db": "neo4j",
        "confidence": "high",
        "rationale": "RBAC junction table — role-to-permission graph is a natural fit.",
        "ddl": """
            CREATE TABLE role_permissions (
                role_id       INTEGER,
                permission_id INTEGER,
                granted_at    TEXT,
                PRIMARY KEY (role_id, permission_id),
                FOREIGN KEY (role_id)       REFERENCES roles(id),
                FOREIGN KEY (permission_id) REFERENCES permissions(id)
            )""",
        "rows": [(1, 101, "2024-01-01"), (1, 102, "2024-01-01"), (2, 101, "2024-02-15")],
    },
    {
        "db_file": "neo4j_05_supplier_products.db",
        "table_name": "supplier_products",
        "expected_db": "neo4j",
        "confidence": "high",
        "rationale": "Many-to-many supplier-to-product supply chain mapping — graph.",
        "ddl": """
            CREATE TABLE supplier_products (
                supplier_id INTEGER,
                product_id  INTEGER,
                lead_days   INTEGER,
                unit_cost   REAL,
                PRIMARY KEY (supplier_id, product_id),
                FOREIGN KEY (supplier_id) REFERENCES suppliers(id),
                FOREIGN KEY (product_id)  REFERENCES products(id)
            )""",
        "rows": [(1, 1, 7, 45.00), (1, 2, 14, 120.00), (2, 1, 5, 48.00)],
    },
    {
        "db_file": "neo4j_06_course_prerequisites.db",
        "table_name": "course_prerequisites",
        "expected_db": "neo4j",
        "confidence": "high",
        "rationale": "DAG of course dependencies — acyclic directed graph, natural for Neo4j.",
        "ddl": """
            CREATE TABLE course_prerequisites (
                course_id     INTEGER,
                prerequisite_id INTEGER,
                PRIMARY KEY (course_id, prerequisite_id),
                FOREIGN KEY (course_id)       REFERENCES courses(id),
                FOREIGN KEY (prerequisite_id) REFERENCES courses(id)
            )""",
        "rows": [(201, 101), (202, 101), (203, 201), (203, 202)],
    },
    {
        "db_file": "neo4j_07_patient_doctors.db",
        "table_name": "patient_doctors",
        "expected_db": "neo4j",
        "confidence": "high",
        "rationale": "Patient-to-doctor assignment — bipartite graph relationship.",
        "ddl": """
            CREATE TABLE patient_doctors (
                patient_id  INTEGER,
                doctor_id   INTEGER,
                assigned_at TEXT,
                is_primary  INTEGER,
                PRIMARY KEY (patient_id, doctor_id),
                FOREIGN KEY (patient_id) REFERENCES patients(id),
                FOREIGN KEY (doctor_id)  REFERENCES doctors(id)
            )""",
        "rows": [(1001, 55, "2023-11-01", 1), (1001, 60, "2024-01-15", 0), (1002, 55, "2024-03-01", 1)],
    },
    {
        "db_file": "neo4j_08_org_hierarchy.db",
        "table_name": "org_hierarchy",
        "expected_db": "neo4j",
        "confidence": "high",
        "rationale": "Self-referencing FK (manager->employee) — tree/hierarchy graph.",
        "ddl": """
            CREATE TABLE org_hierarchy (
                emp_id      INTEGER PRIMARY KEY,
                name        TEXT,
                manager_id  INTEGER,
                level       INTEGER,
                FOREIGN KEY (manager_id) REFERENCES org_hierarchy(emp_id)
            )""",
        "rows": [
            (1, "CEO Alice",    None, 1),
            (2, "VP Bob",       1,    2),
            (3, "Manager Carol",2,    3),
            (4, "Engineer Dave",3,    4),
        ],
    },
    {
        "db_file": "neo4j_09_friend_network.db",
        "table_name": "friendships",
        "expected_db": "neo4j",
        "confidence": "high",
        "rationale": "Symmetric social friendship graph — pure graph edge table.",
        "ddl": """
            CREATE TABLE friendships (
                user_a_id   INTEGER,
                user_b_id   INTEGER,
                since       TEXT,
                PRIMARY KEY (user_a_id, user_b_id),
                FOREIGN KEY (user_a_id) REFERENCES users(id),
                FOREIGN KEY (user_b_id) REFERENCES users(id)
            )""",
        "rows": [(1, 2, "2023-05-01"), (1, 3, "2023-06-15"), (2, 4, "2024-01-10")],
    },
    {
        "db_file": "neo4j_10_tag_assignments.db",
        "table_name": "tag_assignments",
        "expected_db": "neo4j",
        "confidence": "medium",
        "rationale": "Tag-to-resource junction. Medium — could also stay relational if tag filtering via SQL is primary use case.",
        "ddl": """
            CREATE TABLE tag_assignments (
                tag_id      INTEGER,
                resource_id INTEGER,
                resource_type TEXT,
                tagged_at   TEXT,
                PRIMARY KEY (tag_id, resource_id),
                FOREIGN KEY (tag_id) REFERENCES tags(id)
            )""",
        "rows": [
            (1, 100, "post", "2024-01-01"),
            (2, 100, "post", "2024-01-01"),
            (1, 200, "product", "2024-02-10"),
        ],
    },

    # ══════════════════════════════════════════════════════════════════
    # RELATIONAL — transactional / structured / numeric tables (10 cases)
    # ══════════════════════════════════════════════════════════════════

    {
        "db_file": "relational_01_support_tickets.db",
        "table_name": "support_tickets",
        "expected_db": "relational",
        "confidence": "high",
        "rationale": "Transactional record with status/priority enum-like fields — SQL filtering.",
        "ddl": """
            CREATE TABLE support_tickets (
                id           INTEGER PRIMARY KEY,
                requester_id INTEGER,
                issue_summary TEXT,
                status       TEXT,
                priority     TEXT,
                FOREIGN KEY (requester_id) REFERENCES employees(id)
            )""",
        "rows": [
            (1, 104, "Docker container keeps crashing on startup", "Open",     "High"),
            (2, 102, "Need access to the new payroll software",    "Resolved", "Medium"),
        ],
    },
    {
        "db_file": "relational_02_sales_transactions.db",
        "table_name": "sales_transactions",
        "expected_db": "relational",
        "confidence": "high",
        "rationale": "Financial transaction table — numeric aggregation, exact-match, audit trail.",
        "ddl": """
            CREATE TABLE sales_transactions (
                txn_id      INTEGER PRIMARY KEY,
                customer_id INTEGER,
                amount      REAL,
                tax         REAL,
                currency    TEXT,
                status      TEXT,
                created_at  TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            )""",
        "rows": [
            (1001, 5, 1299.99, 103.99, "USD", "completed", "2024-06-01 10:22:00"),
            (1002, 8, 49.95,   3.99,   "USD", "refunded",  "2024-06-02 14:05:00"),
        ],
    },
    {
        "db_file": "relational_03_employee_payroll.db",
        "table_name": "payroll",
        "expected_db": "relational",
        "confidence": "high",
        "rationale": "Payroll record with numeric salary/deduction fields — exact aggregation.",
        "ddl": """
            CREATE TABLE payroll (
                payroll_id  INTEGER PRIMARY KEY,
                employee_id INTEGER,
                period      TEXT,
                gross       REAL,
                deductions  REAL,
                net         REAL,
                paid_at     TEXT,
                FOREIGN KEY (employee_id) REFERENCES employees(id)
            )""",
        "rows": [
            (1, 101, "2024-06", 8500.00, 1200.00, 7300.00, "2024-06-28"),
            (2, 102, "2024-06", 7200.00, 1050.00, 6150.00, "2024-06-28"),
        ],
    },
    {
        "db_file": "relational_04_server_logs.db",
        "table_name": "server_logs",
        "expected_db": "relational",
        "confidence": "high",
        "rationale": "Structured audit/operational log with status codes and timestamps — SQL aggregation.",
        "ddl": """
            CREATE TABLE server_logs (
                log_id      INTEGER PRIMARY KEY,
                server_id   INTEGER,
                event_type  TEXT,
                severity    TEXT,
                http_status INTEGER,
                duration_ms INTEGER,
                logged_at   TEXT
            )""",
        "rows": [
            (1, 10, "REQUEST", "INFO",  200, 45,   "2024-07-01 00:00:01"),
            (2, 10, "REQUEST", "ERROR", 500, 3200, "2024-07-01 00:00:03"),
            (3, 11, "STARTUP", "INFO",  None, None, "2024-07-01 00:00:00"),
        ],
    },
    {
        "db_file": "relational_05_inventory_movements.db",
        "table_name": "inventory_movements",
        "expected_db": "relational",
        "confidence": "high",
        "rationale": "Stock movement ledger — insert-only append log, numeric quantities, SQL SUM.",
        "ddl": """
            CREATE TABLE inventory_movements (
                movement_id INTEGER PRIMARY KEY,
                item_id     INTEGER,
                warehouse   TEXT,
                quantity    INTEGER,
                direction   TEXT,
                moved_at    TEXT,
                reason      TEXT
            )""",
        "rows": [
            (1, 5, "WH-A", 100, "IN",  "2024-06-10", "Purchase Order #2210"),
            (2, 5, "WH-A",  30, "OUT", "2024-06-15", "Customer Order #5501"),
            (3, 7, "WH-B",  50, "IN",  "2024-06-11", "Purchase Order #2211"),
        ],
    },
    {
        "db_file": "relational_06_subscription_plans.db",
        "table_name": "subscription_plans",
        "expected_db": "relational",
        "confidence": "high",
        "rationale": "Pricing/plan lookup table — numeric-heavy, exact-match for billing engine.",
        "ddl": """
            CREATE TABLE subscription_plans (
                plan_id     INTEGER PRIMARY KEY,
                name        TEXT,
                price_month REAL,
                price_year  REAL,
                max_users   INTEGER,
                storage_gb  INTEGER,
                tier        TEXT
            )""",
        "rows": [
            (1, "Starter", 9.99,  99.00,  5,    10, "free"),
            (2, "Pro",     29.99, 299.00, 25,   100, "paid"),
            (3, "Enterprise", 99.99, 999.00, 999, 1000, "paid"),
        ],
    },
    {
        "db_file": "relational_07_leave_requests.db",
        "table_name": "leave_requests",
        "expected_db": "relational",
        "confidence": "high",
        "rationale": "HR operational table — date ranges, status workflow, numeric days — SQL reporting.",
        "ddl": """
            CREATE TABLE leave_requests (
                request_id  INTEGER PRIMARY KEY,
                employee_id INTEGER,
                leave_type  TEXT,
                start_date  TEXT,
                end_date    TEXT,
                days        INTEGER,
                status      TEXT,
                approved_by INTEGER,
                FOREIGN KEY (employee_id)  REFERENCES employees(id),
                FOREIGN KEY (approved_by)  REFERENCES employees(id)
            )""",
        "rows": [
            (1, 101, "Annual",  "2024-07-15", "2024-07-19", 5, "Approved", 102),
            (2, 104, "Sick",    "2024-07-22", "2024-07-22", 1, "Approved", 102),
            (3, 103, "Unpaid",  "2024-08-01", "2024-08-05", 5, "Pending",  None),
        ],
    },
    {
        "db_file": "relational_08_order_items.db",
        "table_name": "order_items",
        "expected_db": "relational",
        "confidence": "high",
        "rationale": "Line-item transactional table — numeric price/qty, JOIN to orders — pure relational.",
        "ddl": """
            CREATE TABLE order_items (
                item_id     INTEGER PRIMARY KEY,
                order_id    INTEGER,
                product_id  INTEGER,
                quantity    INTEGER,
                unit_price  REAL,
                discount    REAL,
                FOREIGN KEY (order_id)   REFERENCES orders(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            )""",
        "rows": [
            (1, 5001, 42, 2, 299.99, 0.00),
            (2, 5001, 17, 1,  49.95, 5.00),
            (3, 5002, 42, 1, 299.99, 15.00),
        ],
    },
    {
        "db_file": "relational_09_attendance.db",
        "table_name": "attendance",
        "expected_db": "relational",
        "confidence": "high",
        "rationale": "Daily attendance timestamps — temporal/numeric, exact-match, GROUP BY reporting.",
        "ddl": """
            CREATE TABLE attendance (
                attendance_id INTEGER PRIMARY KEY,
                employee_id   INTEGER,
                date          TEXT,
                check_in      TEXT,
                check_out     TEXT,
                hours_worked  REAL,
                status        TEXT,
                FOREIGN KEY (employee_id) REFERENCES employees(id)
            )""",
        "rows": [
            (1, 101, "2024-07-01", "09:02", "17:58", 8.93, "Present"),
            (2, 102, "2024-07-01", "08:45", "17:30", 8.75, "Present"),
            (3, 103, "2024-07-01", None,    None,    0.00, "Absent"),
        ],
    },
    {
        "db_file": "relational_10_audit_log.db",
        "table_name": "audit_log",
        "expected_db": "relational",
        "confidence": "high",
        "rationale": "System audit trail — append-only, timestamp-ordered, compliance queries — relational.",
        "ddl": """
            CREATE TABLE audit_log (
                log_id      INTEGER PRIMARY KEY,
                user_id     INTEGER,
                action      TEXT,
                entity_type TEXT,
                entity_id   INTEGER,
                ip_address  TEXT,
                performed_at TEXT
            )""",
        "rows": [
            (1, 101, "UPDATE", "employee", 104, "192.168.1.10", "2024-07-01 09:15:00"),
            (2, 102, "DELETE", "ticket",     2, "10.0.0.5",     "2024-07-01 11:30:00"),
            (3, 101, "CREATE", "project",    3, "192.168.1.10", "2024-07-01 14:00:00"),
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Build each .db file
# ─────────────────────────────────────────────────────────────────────────────
def create_all_dbs():
    ground_truth = []

    for entry in EVAL_TABLES:
        db_path = os.path.join(OUTPUT_DIR, entry["db_file"])

        # Remove stale db
        if os.path.exists(db_path):
            os.remove(db_path)

        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute(entry["ddl"])

        if entry["rows"]:
            placeholders = ", ".join("?" * len(entry["rows"][0]))
            cur.executemany(
                f"INSERT OR IGNORE INTO {entry['table_name']} VALUES ({placeholders})",
                entry["rows"],
            )

        conn.commit()
        conn.close()

        ground_truth.append({
            "db_file":     entry["db_file"],
            "table_name":  entry["table_name"],
            "expected_db": entry["expected_db"],
            "confidence":  entry["confidence"],
            "rationale":   entry["rationale"],
        })

        print(f"  ✓  {entry['db_file']:45s}  [{entry['expected_db'].upper():10s}]  ({entry['confidence']})")

    # Write ground_truth.json
    with open(GT_PATH, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nGround truth written to: {GT_PATH}")
    print(f"Databases written to:    {OUTPUT_DIR}/")
    print(f"Total: {len(EVAL_TABLES)} databases")


if __name__ == "__main__":
    print("Creating evaluation databases...\n")
    create_all_dbs()
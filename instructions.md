# Data Migration Setup & Run
---

## Create a SurrealDB Instance

* Go to SurrealDB Cloud
* Create a new instance
* Choose region (e.g., `aws-aps1`)
* Copy your instance URL

Example:

```
wss://your-instance-name.aws-aps1.surreal.cloud
```

---

## Configure Database User

Run the following query in the SurrealDB SQL editor:

```sql
DEFINE USER admin ON ROOT PASSWORD 'MyPassword123' ROLES OWNER;
```

---

## Install Dependencies

In your VS Code terminal, run:

```bash
pip install surrealdb sqlalchemy langgraph
```

---

## Run the Migration Pipeline

Execute the following command:

```bash
python -m core.pipeline \
  --db sqlite:///enterprise_system.db \
  --surreal-url wss://your-instance-name.aws-aps1.surreal.cloud \
  --surreal-ns main \
  --surreal-db main \
  --surreal-user admin \
  --surreal-pass MyPassword123
```

---

## Verify Migration

* Open your SurrealDB dashboard
* Navigate to your database (`main`)
* Confirm records are present

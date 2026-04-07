import chromadb
from pymongo import MongoClient
from neo4j import GraphDatabase
from sqlalchemy import create_engine, text

def audit_databases():
    print("POST-MIGRATION AUDIT")

    # 1. CHROMA (Vector Search Test)
    try:
        print("CHROMA DB")
        client = chromadb.PersistentClient(path="./chroma_store")
        # Replace 'knowledge_base' with whatever table went to Chroma
        collection = client.get_collection(name="knowledge_base") 
        results = collection.query(query_texts=["testing connection"], n_results=1)
        print(f"Chroma Connection Active. Found Document: {results['documents'][0]}")
    except Exception as e:
        print(f"Chroma Error: {e}")

    # 2. MONGO (Document Denormalization Test)
    try:
        print("\nMONGO DB")
        mongo_client = MongoClient("mongodb://localhost:27017")
        db = mongo_client["polyglot_migration"]
        # Replace 'projects' with your Mongo table
        doc = db["projects"].find_one() 
        print(f"Mongo Connection Active. Sample Document:\n{doc}")
    except Exception as e:
        print(f"Mongo Error: {e}")

    # 3. NEO4J (Graph Traversal Test)
    try:
        print("\nNEO4J")
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "test1234")) # Update password!
        with driver.session() as session:
            # Simple query to find one node and its relationships
            result = session.run("MATCH (n)-[r]->(m) RETURN labels(n)[0] AS label, type(r) AS rel LIMIT 1")
            record = result.single()
            if record:
                print(f"Neo4j Connection Active. Found Edge: ({record['label']}) -[{record['rel']}]-> (...)")
            else:
                print("Neo4j Active, but no relationships found yet.")
    except Exception as e:
        print(f"Neo4j Error: {e}")

    # 4. MYSQL (Relational Test)
    try:
        print("\nMYSQL")
        import urllib.parse
        encoded_pwd = urllib.parse.quote_plus("your pwd!!!!")
        
        engine = create_engine(f"mysql+pymysql://root:{encoded_pwd}@localhost:3306/polyglot_target")
        with engine.connect() as conn:
            result = conn.execute(text("SHOW TABLES;")).fetchall()
            print(f"MySQL Connection Active. Tables found: {[r[0] for r in result]}")
    except Exception as e:
        print(f"MySQL Error: {e}")
if __name__ == "__main__":
    audit_databases()
import pandas as pd
from vespa.application import Vespa
from vespa.io import VespaQueryResponse, VespaResponse

# Function to display hits as a DataFrame
def display_hits_as_df(response: VespaQueryResponse, fields) -> pd.DataFrame:
    records = []
    for hit in response.hits:
        record = {}
        for field in fields:
            record[field] = hit["fields"][field]
        records.append(record)
    return pd.DataFrame(records)

# Keyword Search: Search using "Harry Potter"
def keyword_search(app, search_query):
    query = {
        "yql": "select * from sources * where userQuery() limit 5",
        "query": search_query,
        "ranking": "bm25",
    }
    response = app.query(query)
    return display_hits_as_df(response, ["doc_id", "title"])

# Semantic Search: Search using "Harry Potter"
def semantic_search(app, query):
    query = {
        "yql": "select * from sources * where ({targetHits:100}nearestNeighbor(embedding,e)) limit 5",
        "query": query,
        "ranking": "semantic",
        "input.query(e)": "embed(@query)"
    }
    response = app.query(query)
    return display_hits_as_df(response, ["doc_id", "title"])

# Get the embedding of a document by doc_id
def get_embedding(doc_id):
    query = {
        "yql" : f"select doc_id, title, text, embedding from content.doc where doc_id contains '{doc_id}'",
        "hits": 1
    }
    result = app.query(query)
    
    if result.hits:
        return result.hits[0]
    return None

# Query movies by embedding: Perform recommendation search
def query_movies_by_embedding(embedding_vector):
    query = {
        'hits': 5,
        'yql': 'select * from content.doc where ({targetHits:5}nearestNeighbor(embedding, user_embedding))',
        'ranking.features.query(user_embedding)': str(embedding_vector),
        'ranking.profile': 'recommendation'
    }
    return app.query(query)

# Replace with the host and port of your local Vespa instance
app = Vespa(url="http://localhost", port=8080)

# Query for keyword and semantic search using "Harry Potter"
query = "Harry Potter"

# Perform Keyword Search
print("Running Keyword Search for:", query)
df = keyword_search(app, query)
print(df.head())

# Perform Semantic Search
print("\nRunning Semantic Search for:", query)
df = semantic_search(app, query)
print(df.head())

# Get embedding for doc_id=559
doc_id = 559
print(f"\nGetting embedding for doc_id {doc_id}")
emb = get_embedding(doc_id)
if emb:
    # Perform recommendation search
    print(f"\nRunning Recommendation Search for doc_id {doc_id}")
    results = query_movies_by_embedding(emb["fields"]["embedding"])
    df = display_hits_as_df(results, ["doc_id", "title", "text"])
    print(df.head())
else:
    print(f"Document {doc_id} not found.")

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64, Float32, Array

# Define your entity (primary key for feature lookup)
cat_fact = Entity(name="cat_fact", join_keys=["cat_fact_id"])

# Define offline source 
cat_facts_source = FileSource(
    name="cat_facts_source",
    path="data/cat_facts.parquet",
    timestamp_field="event_timestamp",
)

# Define feature view for embeddings
cat_facts_fv = FeatureView(
    name="cat_facts_embeddings",
    entities=[cat_fact],
    ttl=timedelta(days=1),
    schema=[
        Field(
            name="vector",
            dtype=Array(Float32),
            vector_index=True,
            vector_search_metric="COSINE"
        ),
    ],
    online=True,
    source=cat_facts_source,
)

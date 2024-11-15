import os
from typing import List
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime
import pickle
import pandas as pd
from sqlalchemy import create_engine
from fastapi import HTTPException

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def save_model(model, path: str):
    model_path = get_model_path(path)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def load_models():
    model_path = get_model_path("model_pipeline.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def batch_load_sql(query: str) -> pd.DataFrame:
    global engine
    CHUNKSIZE = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_features(query) -> pd.DataFrame:
    return batch_load_sql(query)

app = FastAPI()

engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )


model = load_models()
print("Model loaded")

# Load user data
query = "SELECT * FROM ivan_golov_features_lesson_22"
user_data = load_features(query)

# Load post data
post_text_query = "SELECT * FROM public.post_text_df"
post_text_df = load_features(post_text_query)

# Load the unique user_id, post_id pairs
query_test = "SELECT DISTINCT user_id, post_id FROM public.feed_data WHERE action = 'like';"
test_df = load_features(query_test)

print("Data loaded")

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    # Filter user data for the given user_id
    user_data_filtered = user_data[user_data['user_id'] == id]
    
    # Filter test data for the given user_id
    test_data_filtered = test_df[test_df['user_id'] == id]
    
    # Check if the number of rows is less than 20
    if len(test_data_filtered) < 20:
        # Randomly sample rows to fill up to 20
        additional_rows = test_df.sample(n=20 - len(test_data_filtered), replace=True)
        test_data_filtered = pd.concat([test_data_filtered, additional_rows])
    
    # Merge with user_data_df by user_id
    merged_data = pd.merge(test_data_filtered, user_data_filtered , on='user_id')
    
    # Merge with post_text_df by post_id
    merged_data = pd.merge(merged_data, post_text_df, on='post_id')
    
    # Remove user_id and post_id columns before making the inference
    inference_data = merged_data.drop(columns=['user_id', 'post_id'])

    # Perform inference using the model
    predictions = model.predict_proba(inference_data)[:, 1]

    # Add predictions to the original dataframe
    merged_data['prediction'] = predictions
    
    # Sort by prediction and take the top LIMIT
    top_posts = merged_data.sort_values(by='prediction', ascending=False).head(limit)
        
    # Prepare the response
    response = [
        PostGet(id=row['post_id'], text=row['text'], topic=row['topic'])
        for _, row in top_posts.iterrows()
    ]
    
    if not response:
        raise HTTPException(status_code=404, detail="No recommendations found")
    
    return response

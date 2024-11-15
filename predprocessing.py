import pandas as pd
from sqlalchemy import create_engine

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def extract_features():
    # Создаем соединение с базой данных
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )

    # Загрузка данных из таблицы user_data
    user_data_query = "SELECT * FROM public.user_data"
    user_data = pd.read_sql(user_data_query, engine)

    # Преобразования и фича инжиниринг
    sng_countries = user_data['country'].value_counts().index[:4].to_list()
    user_data['sng_country'] = user_data['country'].apply(lambda x: 1 if x in sng_countries else 0)
    user_data = user_data.drop('country', axis=1)
    
    top_50_cities = user_data['city'].value_counts().index[:50].to_list()
    user_data['top_50_cities'] = user_data['city'].apply(lambda x: 1 if x in top_50_cities else 0)
    user_data = user_data.drop('city', axis=1)
    
    for col in ['os', 'source']:
        one_hot = pd.get_dummies(user_data[col],prefix=col,drop_first=True)
        user_data = pd.concat((user_data.drop(col,axis=1),one_hot.astype(int)),axis=1)
    
    active_query = """
    SELECT 
        user_id,
        SUM(CASE WHEN action = 'like' THEN 1 ELSE 0 END) AS number_of_likes
    FROM public.feed_data
    GROUP BY user_id
    """
    active = pd.read_sql(active_query, engine)
    user_data = user_data.merge(active, on='user_id', how='left')
    
    user_data['active'] = user_data['number_of_likes'].apply(lambda x: 1 if x > 26 else 0)
    user_data = user_data.drop('number_of_likes', axis=1)

    # Запись объединенных данных в новую таблицу
    table_name = 'ivan_golov_features_lesson_22'
    user_data.to_sql(table_name, con=engine, if_exists='replace', index=False)

def load_features() -> pd.DataFrame:
    query = "SELECT * FROM ivan_golov_features_lesson_22"
    return batch_load_sql(query)

if __name__ == "__main__":
    extract_features()
    features = load_features()
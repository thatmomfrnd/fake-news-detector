import pandas as pd
from sklearn.model_selection import train_test_split
def load_data(path):
    df=pd.read_csv(path)
    df=df[['title','text','label']]
    df=df.dropna()
    df['content']=df['title']+" "+df['text']
    return df[['content','label']] 

def prep_data(df,test_size=0.2):
    x=df['content'].tolist()
    y=df['label'].tolist()
    return train_test_split(
        x,y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

if __name__ == "__main__":
    df = load_data("data/news.csv")

    x_train, x_test, y_train, y_test = prep_data(df)
    print("Training samples:", len(x_train))
    print("Testing samples:", len(x_test))
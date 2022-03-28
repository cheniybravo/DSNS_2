import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify

from plotly.graph_objs import Bar, Layout, Figure
import plotly.express as px
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    
    # Detect & Replace URL
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_regex, text)
    for url in urls:
        text = text.replace(url, "urlplaceholder")
        
    # normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    
    clean_words = []
    for word in words:
        clean_word = lemmatizer.lemmatize(word).strip()
        clean_words.append(clean_word)
        
    return clean_words


# load data
engine = create_engine('sqlite:///../data/MyEngine_Yi.db')
df = pd.read_sql_table('RawData', engine)

# load model
model = joblib.load("../models/NLP_model_Yi.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')


def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    summ_category = pd.DataFrame(df.iloc[:,4:].sum(), columns = ['mess_volume']).reset_index().rename(columns={'index':'Category'})
    summ_category['rank'] = summ_category['mess_volume'].rank(method = 'dense', ascending = False)
    summ_category.loc[summ_category['rank'] > 10, 'Category'] = "Other Categories"
    
    df_cut = df.iloc[:, 3:]
    df_sum = df_cut.groupby('genre').agg(lambda x: x.sum()/x.shape[0]).transpose().reset_index().rename(columns = {'index': 'Category'})
    df_sum2 = pd.melt(df_sum, id_vars=['Category'], value_vars=['direct', 'news', 'social'], value_name = 'event_rate')
    df_size = df_cut.groupby('genre').agg(lambda x: x.shape[0]).transpose().reset_index().rename(columns = {'index': 'Category'})
    df_size2 = pd.melt(df_size, id_vars=['Category'], value_vars=['direct', 'news', 'social'], value_name = 'Volume')
    df_all = pd.merge(df_sum2, df_size2, how = 'inner', on = ['Category', 'genre'])

    # create visuals
    trace1 = [Bar(x=genre_names,
                  y=genre_counts)]

    layout1 = Layout(title= 'Distribution of Message Genres',
                     yaxis= {'title': "Count"},
                     xaxis= {'title': "Genre"})
    fig1 = Figure(data = trace1, layout = layout1)
    
    fig2 = px.pie(summ_category, values='mess_volume', names='Category', title= 'Top 10 Message Volume Categories of All Genres')
    
    fig3 = px.scatter(df_all, x="Category", y="Volume", color="genre", size = "event_rate", hover_data=['event_rate'],title= "Message Volume/Event Rate by Categories")
    
    graphs = [fig1, fig2, fig3]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
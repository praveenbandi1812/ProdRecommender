# import Flask class from the flask module
#from flask import Flask, request
from flask import Flask, jsonify,  request, render_template
import joblib
import numpy as np
#from sklearn.externals 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Create Flask object to run
app = Flask(__name__)

# Load the model from the file
#iris_model = joblib.load('model/iris_model.pkl')
xgb_model_pk = joblib.load('xgb_model.pkl')
item_final_rating_pk = joblib.load('item_final_rating.pkl')
Product_Reviews = joblib.load('Product_Reviews_CleanData.pkl')
word_vectorizer = joblib.load('word_vectorizer.pkl')
Item_Reviews = joblib.load('Item_Reviews.pkl')
#Product_Reviews = pd.read_csv("sample30.csv")
#Item_Reviews=Product_Reviews
#Item_Reviews.drop_duplicates(subset =["reviews_username","name","id"],keep = False, inplace = True)
    # define vectorize and fit to data     
#word_vectorizer = TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',stop_words='english',ngram_range=(1, 1),max_features=10000)
#word_vectorizer.fit(Product_Reviews['reviews_text'])

@app.route('/')
def home():
    #return "Sentiment ier"
    return render_template('index.html')

@app.route('/predict')
def predict():
    # Get values from browser
    #sepal_length = request.args['sepal_length']
    #sepal_width = request.args['sepal_width']
    #petal_length = request.args['petal_length']
    #petal_width = request.args['petal_width']
    user_input = request.args['username']
    
    #print(sepal_length)
    #Top20_Recomm_products=item_final_rating_pk.loc[user_input].sort_values(ascending=False)[0:20]
    Top20_Recomm_products_df = pd.DataFrame(item_final_rating_pk.loc[user_input].sort_values(ascending=False)[0:20])
    ##### merging with main dataset and getting reviews for the above 20 products
    Top20_Prod_reviews =    Item_Reviews[Item_Reviews['name'].isin(Top20_Recomm_products_df.index)]
    #Top20_Prod_reviews.insert(15, "Sentiscore", 0 )
    #for index, row in Top20_Prod_reviews.iterrows():
     #   review_str = Top20_Prod_reviews.at[index,'reviews_text']
        #Result=self.sentimental_Score(review_str)
        #test_inp = review_str.split('.')
        #Result = xgb_model_pk.predict(word_vectorizer.transform(test_inp))
      #  Result = xgb_model_pk.predict(word_vectorizer.transform(review_str.split('.')))
       # Top20_Prod_reviews.at[index,'Sentiscore'] = Result[0]
        #Top20_Prod_reviews.at[index,'Sentiscore'] = xgb_model_pk.predict(word_vectorizer.transform(test_inp))
    Top5_recommended_products=Top20_Prod_reviews.groupby(['name']).sum().sort_values(by='user_sentiment',ascending=False)[0:5]

    #output = "Predicted Iris Class: " + str(class_predicted)
    output = "TOP 5 Recommended Products are : " + str(Top5_recommended_products)
    #output = "TOP 5 Recommended Products are : " + Top5_recommended_products
    #output = "Sentiment for the given sentence is: "

    #return (output)
    return render_template('index.html', prediction_text='Output {}'.format(output))


if __name__ == "__main__":
    # Start Application
    app.run()
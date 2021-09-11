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

xgb_model_pk = joblib.load('xgb_model.pkl')
item_final_rating_pk = joblib.load('item_final_rating.pkl')
Product_Reviews = joblib.load('Product_Reviews_CleanData.pkl')
word_vectorizer = joblib.load('word_vectorizer.pkl')
Item_Reviews = joblib.load('Item_Reviews.pkl')

@app.route('/')
def home():
    #return "Sentiment ier"
    return render_template('index.html')

@app.route('/predict')
def predict():
    user_input = request.args['username']
    #Top20_Recomm_products=item_final_rating_pk.loc[user_input].sort_values(ascending=False)[0:20]
    try:
        Top20_Recomm_products_df = pd.DataFrame(item_final_rating_pk.loc[user_input].sort_values(ascending=False)[0:20])
        ##### merging with main dataset and getting reviews for the above 20 products
        Top20_Prod_reviews =    Item_Reviews[Item_Reviews['name'].isin(Top20_Recomm_products_df.index)]
        Top5_recommended_products=Top20_Prod_reviews.groupby(['name']).sum().sort_values(by='user_sentiment',ascending=False)[0:5]
        Top5_recommended_products_df = pd.DataFrame(Top5_recommended_products.index)
        output2 = Top5_recommended_products_df.to_string(header=False,index=False)
        output= output2.split("\n")
    except Exception as e:
        output = "Sorry, we are unable to predict for the given user"
    #return render_template('index.html', header='TOP 5 Recommended Products', prediction_text='Recommended Products are - {}'.format(output))
    return render_template('index.html', header='TOP 5 Recommended Products', prediction_text=output)
    #return render_template('index.html',header='Amazing Universe', sub_header='Our universe is quite amazing', list_header="Galaxies!",prediction_text=output, site_title="Camposha.info")
if __name__ == "__main__":
    app.run()
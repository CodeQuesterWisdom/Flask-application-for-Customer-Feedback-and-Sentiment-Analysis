import pandas
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import tweepy   # To consume Twitter's API  # For number computing
from textblob import TextBlob
import os
from flask import Flask, request, render_template, jsonify, Markup
import matplotlib.pyplot as plt
from collections import Counter
import csv
import numpy
import operator
from datetime import datetime
import folium
import geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import nltk
nltk.download()
import collections
from nltk.corpus import stopwords
stemmer=nltk.PorterStemmer()
stops = set(stopwords.words("english"))


#step1: reading data
train_data= pandas.read_csv("Sentiment Analysis Dataset.csv",header=0,error_bad_lines=False)

preprocessed_words=[]
#step2: Data preprocessing
def preprocessing(raw_data):
    review_text = BeautifulSoup(raw_data,"html5lib").get_text() #removes html tags
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    meaningful_words = list(filter(lambda x: (stemmer.stem(x)) , words))
    meaningful_words = [w for w in meaningful_words if not w in stops]
    for wo in meaningful_words:
        if not 2<len(wo)<31:
            meaningful_words.remove(wo)
    for we in meaningful_words:       #optimisation required
        preprocessed_words.append(we)
    return( " ".join(meaningful_words ))


clean_train_reviews = []

#num_reviews = train_data["Sentiment"].size
print ("Cleaning and parsing the train set movie reviews...")


#increase number from 500 to your value for accurate results
for i in range( 0, 500):
    clean_train_reviews.append(preprocessing(train_data["SentimentText"][i] ) )


final_words_count=collections.Counter(preprocessed_words)
final_words=[]
for letter,count in final_words_count.most_common(1000):     #optimisation required
    final_words.append(letter)


preprocessed_words=[]
#print(clean_train_reviews)

#step3: feautureVectors
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

train_data_features = CountVectorizer(vocabulary=final_words)
train_data_features=train_data_features.fit_transform(clean_train_reviews).toarray()


#step4: Building model
forest = RandomForestClassifier(n_estimators = 150)
forest = forest.fit( train_data_features, train_data["Sentiment"][0:500] )



#Give Credentials
consumer_key =""
consumer_secret=""

access_token=""
access_token_secret=""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Return API with authentication:
api = tweepy.API(auth)


app = Flask(__name__)

@app.route('/index',methods=['GET','POST'])
def index():

    if request.method=="POST":
        topic=request.form['topic']
        test_data=api.search(topic,count =100)

        clean_test_reviews=[]
        user_name=[]
        loc=[]
        timezone=[]
        creationdatee=[]
        polarity=[]
        image=[]
        follower_count=[]
        verified_check=[]
        following_count=[]
        retweets_count=[]
        favourite_count=[]
        tweet_id=[]


        def truncate(f, n):
            '''Truncates/pads a float f to n decimal places without rounding'''
            s = '{}'.format(f)
            if 'e' in s or 'E' in s:
                return '{0:.{1}f}'.format(f, n)
            i, p, d = s.partition('.')
            return '.'.join([i, (d+'0'*n)[:n]])
			
	preprocessed_words=[]
	clean_test_reviews = []
        
        def preprocessing(raw_data):
            review_text = BeautifulSoup(raw_data,"html5lib").get_text() #removes html tags
            letters_only = re.sub("[^a-zA-Z]", " ", review_text)
            words = letters_only.lower().split()
            meaningful_words = list(filter(lambda x: (stemmer.stem(x)) , words))
            meaningful_words = [w for w in meaningful_words if not w in stops]
            for wo in meaningful_words:
                if not 2<len(wo)<31:
                    meaningful_words.remove(wo)
            for we in meaningful_words:       #optimisation required
                preprocessed_words.append(we)
			return(" ".join(meaningful_words ))
			
		
	for tweet in test_data:
			
		clean_test_reviews.append(preprocessing(tweet.text))


        top_words1=[]

        final_words_count=collections.Counter(preprocessed_words)

        for letter,count in final_words_count.most_common(30):     #optimisation required
            top_words1.append(letter)

        top_words=top_words1[10:30]
			
	
	
	test_data_features = CountVectorizer(vocabulary=final_words)
	test_data_features= test_data_features.fit_transform(clean_test_reviews).toarray()
	result = forest.predict(test_data_features)
		

        for tweet in test_data:

            user_name.append(tweet.user.screen_name)
            loc.append(tweet.user.location)
            timezone.append(tweet.user.time_zone)
            creationdatee.append(tweet.created_at)
            polarity.append(truncate((TextBlob(tweet.text).sentiment.polarity),3))
            image.append(tweet.user.profile_image_url_https)
            follower_count.append(tweet.user.followers_count)
            verified_check.append(tweet.user.verified)
            following_count.append(tweet.user.friends_count)
            retweets_count.append(tweet.retweet_count)
            favourite_count.append(tweet.favorite_count)
            tweet_id.append(tweet.id)


        creationdate=[]
        for i in creationdatee:
            creationdate.append(i.strftime('%m/%d/%Y %I:%M:%S'))

        loca=[]
        for i in loc:
            loca.append( re.sub("[^a-zA-Z]", "", i))


        location=[]
        for l in loca:
            if not len(l)>0:
                location.append(l.replace(l,'delhi'))
            else:
                location.append(l)

        for tweet in test_data:
            review_text = BeautifulSoup(tweet.text,"html5lib").get_text()
            letters_only = re.sub("[^a-zA-Z]", " ", review_text)
            clean_test_reviews.append( letters_only)

       


        output = pandas.DataFrame(data={"SentimentText":clean_test_reviews[0:100],"Sentiment":result[0:100],"UserName":user_name[0:100],"Polarity":polarity[0:100],"CreationDate":creationdate[0:100],"Image":image[0:100],"Location":location[0:100],"FollowerCount":follower_count[0:100],"FollowingCount":following_count[0:100],
        "Verified":verified_check[0:100],"ReTweet":retweets_count[0:100], "Likes" : favourite_count[0:100],"ID":tweet_id[0:100] } )

        # Use pandas to write the comma-separated output file
        output.to_csv( "Entire_Output.csv", index=False)

        with open("Entire_Output.csv",newline='') as csvfile:
            spamreader = csv.DictReader(csvfile)
            sortedlist = sorted(spamreader, key=lambda row:(row['Polarity']), reverse=True)


        with open('Sorted_Entire_Output.csv', 'w') as f:
            fieldnames = ['SentimentText','Sentiment', 'UserName', 'Polarity','CreationDate','Image','Location',"FollowerCount","FollowingCount","Verified","ReTweet","Likes","ID"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sortedlist:
                writer.writerow(row)



        with open("Sorted_Entire_Output.csv",newline='') as f0:
            reader = csv.DictReader(f0)
            rows0 = [row for row in reader if float(row['Polarity']) > 0]


        with open('Positive_Output.csv', 'w') as f:
            fieldnames = ['SentimentText','Sentiment', 'UserName', 'Polarity','CreationDate','Image','Location',"FollowerCount","FollowingCount","Verified","ReTweet","Likes","ID"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows0:
                writer.writerow(row)

        with open("Sorted_Entire_Output.csv",newline='') as f0:
            reader = csv.DictReader(f0)
            rows1 = [row for row in reader if float(row['Polarity']) < 0]

        with open('Negative_Output.csv', 'w') as f:
            fieldnames = ['SentimentText','Sentiment',  'UserName', 'Polarity','CreationDate','Image','Location',"FollowerCount","FollowingCount","Verified","ReTweet","Likes","ID"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows1:
                writer.writerow(row)

        with open("Sorted_Entire_Output.csv",newline='') as f0:
            reader = csv.DictReader(f0)
            rows2 = [row for row in reader if float(row['Polarity']) ==0.0]

        with open('Neutral_Output.csv', 'w') as f:
            fieldnames = ['SentimentText','Sentiment', 'UserName', 'Polarity','CreationDate','Image','Location',"FollowerCount","FollowingCount","Verified","ReTweet","Likes","ID"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows2:
                writer.writerow(row)


        df0 = pandas.read_csv("Positive_Output.csv")
        sentiment_text0 = df0['SentimentText']
        user_name0 = df0['UserName']
        polarity0=df0['Polarity']
        creationdate0=df0['CreationDate']
        image0=df0['Image']
        follower_count0=df0["FollowerCount"]
        following_count0=df0["FollowingCount"]
        verified0=df0['Verified']
        retweets_count0=df0['ReTweet']
        favourite_count0=df0['Likes']
        tweet_id0=df0['ID']
        row0,column0=df0.shape

        df1 = pandas.read_csv("Negative_Output.csv")
        sentiment_text1 = df1['SentimentText']
        user_name1= df1['UserName']
        polarity1=df1['Polarity']
        creationdate1=df1['CreationDate']
        image1=df1['Image']
        follower_count1=df1["FollowerCount"]
        following_count1=df1["FollowingCount"]
        verified1=df1['Verified']
        retweets_count1=df1['ReTweet']
        favourite_count1=df1['Likes']
        tweet_id1=df1['ID']
        row1,column1=df1.shape


        df2 = pandas.read_csv("Neutral_Output.csv")
        sentiment_text2 = df2['SentimentText']
        user_name2 = df2['UserName']
        polarity2=df2['Polarity']
        creationdate2=df2['CreationDate']
        image2=df2['Image']
        follower_count2=df2["FollowerCount"]
        following_count2=df2["FollowingCount"]
        verified2=df2['Verified']
        retweets_count2=df2['ReTweet']
        favourite_count2=df2['Likes']
        tweet_id2=df2['ID']
        row2,column2=df2.shape

        #with open("Sorted_Entire_Output.csv", 'r') as data:
         #  counter = Counter()
          # for row in csv.DictReader(data):
            #   counter[row['Sentiment']] += 1

        positive = row0
        negative = row1
        neutral =  row2

        labels=["Positive","Negative","Neutral"]
        values=[positive,negative,neutral]
        colors=["#70db70"," #ff6666","#4dffff"]


        return render_template('index.html',topic=topic,set=zip(values, labels, colors),values=values,colors=colors, labels=labels,sentiment_text1=sentiment_text1,sentiment_text0=sentiment_text0,sentiment_text2=sentiment_text2,
        user_name1=user_name1,user_name0=user_name0,user_name2=user_name2,polarity1=polarity1,polarity0=polarity0,polarity2=polarity2,creationdate1=creationdate1,creationdate0=creationdate0,
        creationdate2=creationdate2,image0=image0,image1=image1,image2=image2,pol="Polarity : ",follower_count0=follower_count0,follower_count1=follower_count1,follower_count2=follower_count2,
        following_count0=following_count0,following_count1=following_count1,following_count2=following_count2,verified0=verified0,verified1=verified1,verified2=verified2,followers="Followers : ",following="Following : ",
        retweets_count0=retweets_count0,retweets_count1=retweets_count1,retweets_count2=retweets_count2,favourite_count0=favourite_count0,favourite_count1=favourite_count1,favourite_count2=favourite_count2,
        tweet_id0=tweet_id0,tweet_id1=tweet_id1,tweet_id2=tweet_id2,top_words=top_words)

    return render_template('index.html')


@app.route('/')
def parallax():

    return render_template('parallax.html')

@app.route('/credentials',methods=['GET','POST'])
def credentials():
    password=request.form['password']
    if(password=="jpmc"):
        return render_template('index.html')
    return('Wrong Access Code')




@app.route('/data') # this is a job for GET, not POST
def data():
    return send_file('Entire_Output.csv',
                     mimetype='text/csv',
                     attachment_filename='data.csv',
                     as_attachment=True)


@app.route('/map1', methods=['GET','POST'])
def map1():
    map=folium.Map(location=[38.58,-99.09],zoom_start=2,tiles="Mapbox Bright")
    df_map = pandas.read_csv("Entire_Output.csv")
    location_map = df_map['Location']
    data=df_map['SentimentText']
    polarity_map=df_map['Polarity']
    fg0=folium.FeatureGroup(name="Positive")
    fg1=folium.FeatureGroup(name="Negative")
    fg2=folium.FeatureGroup(name="Neutral")
    for loc_map,data_map,pol_map in zip(location_map,data,polarity_map):

        if not loc_map=="delhi":
            try:
                geolocator = Nominatim()
                locationn = geolocator.geocode(loc_map)
                addr=locationn.address
                finallat=locationn.latitude
                finallon=locationn.longitude
                if pol_map>0:
                    fg0.add_child(folium.Marker(location=[finallat,finallon],popup=str(loc_map)+ "\n..............."+str(data_map),icon=folium.Icon(color='green')))
                elif pol_map<0:
                    fg1.add_child(folium.Marker(location=[finallat,finallon],popup=str(loc_map)+"\n..............."+str(data_map),icon=folium.Icon(color='red')))
                else:
                    fg2.add_child(folium.Marker(location=[finallat,finallon],popup=str(loc_map)+"\n................"+str(data_map),icon=folium.Icon(color='blue')))
            except AttributeError:
                print("Problem with data or cannot Geocode.")
            except GeocoderTimedOut:
                print("Time out ")

    map.add_child(fg0)
    map.add_child(fg1)
    map.add_child(fg2)
    map.add_child(folium.LayerControl())
    map.save("templates/map1.html")

    return render_template('map1.html')

if __name__ == '__main__':
   app.run(debug = True)

#port = int(os.environ.get('PORT', 5000))
#app.run(host="127.0.0.1", port=port, debug=True)

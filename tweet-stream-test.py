import tweepy
from tweepy import Stream
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
 
consumer_key = 'QMVyK39ZrZ7G6q8qECKx9zHjx'
consumer_secret = 'iOfdLzfdceYqZeI4XlbZPXruqSc1pWpA3hmcTCVpgLOlamvBDG'
access_token = '244792775-pQrpGYOLYn5OYW19Rv19X6hRC1zMNkwEzJZg13Kc'
access_secret = 'Ex7k9unZzrQ69NQvyhMJQkDvreWErkpzvNM5nGrQRCYqS'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

class MyListener(StreamListener):
 
    def on_data(self, data):
        try:
            with open('happy.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        return True
 
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=['#happy'])
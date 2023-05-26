import urllib.request

# Download the pre-trained Cascade Classifier for traffic lights
url = 'https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_traffic_signal.xml'
urllib.request.urlretrieve(url, 'traffic_light_cascade.xml')
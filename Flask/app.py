import pickle
import cv2
from skimage import feature
from flask import Flask, request, render_template, redirect, url_for
from functools import wraps
import os.path

app = Flask(__name__)
app.secret_key = 'secretkey'

# create username and password dictionary
users = {
    "user1": "password1",
    "user2": "password2"
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' in session and 'password' in session:
            if session['username'] in users and session['password'] == users[session['username']]:
                return f(*args, **kwargs)
        return redirect(url_for('login', next=request.url))
    return decorated_function

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session.pop('username', None)
        session.pop('password', None)
        username = request.form['username']
        password = request.form['password']
        if username in users and password == users[username]:
            session['username'] = username
            session['password'] = password
            return redirect(url_for('protected'))
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/protected')
@login_required
def protected():
    return render_template('protected.html')

@app.route('/predict',methods=['GET','POST'])
@login_required
def upload():
    if request.method == 'POST':
        f = request.files['img']
        basepath = os.path.dirname(__file__)
        print(basepath)
        filepath = os.path.join(basepath,"uploads",f.filename)
        f.save(filepath)
        
        print("[INFO] Loading the model...")
        model = pickle.loads(open('parkPredict.pkl',"rb").read())
        image = cv2.imread(filepath)
        output = image.copy()
        
        output = cv2.resize(output,(128,128))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(200,200))
        image = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        
        features = feature.hog(image,orientations=9, pixels_per_cell=(10,10),cells_per_block=(2,2),transform_sqrt=True,block_norm = "L1")
        preds = model.predict([features])
        print(preds)
        
        ls = ["healthy","parkinsons"]
        result = ls[preds[0]]
        
        
        color = (0,255,0) if result == "healthy" else (0,0,255)
        cv2.putText(output,result,(3,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        cv2.imshow("OUTPUT",output)
        cv2.waitKey(0)
        return result
    else:
        print("FAILED...")
    return None

if __name__ == "__main__":
    app.run(debug=True)

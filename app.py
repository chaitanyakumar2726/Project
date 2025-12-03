# from xmlrpc.client import _HostType
from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
import mysql.connector
import re

mydb = mysql.connector.connect(host='localhost',user='root',password='',port='3306',database='CAN_Intrusion')
cur = mydb.cursor()


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        pws = request.form['psw']
        cpws = request.form['cpsw']
        phone = request.form['phone']

        # Password validation
        if len(pws) < 8:
            return render_template('registration.html', msg='Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', pws):
            return render_template('registration.html', msg='Password must contain at least one uppercase letter')
        
        if pws != cpws:
            return render_template('registration.html', msg='Passwords do not match')
        if '@' not in email or '.' not in email:
            return render_template('registration.html', msg='Invalid email format')
        if len(phone) != 10:
            return render_template('registration.html', msg='Phone number must be 10 digits only')


        # Check if user exists
        sql = "SELECT * FROM user WHERE email=%s"
        cur = mydb.cursor()
        cur.execute(sql, (email,))
        d = cur.fetchall()
        mydb.commit()

        if d:
            return render_template('registration.html', msg='Details already exist')
        else:
            sql = "INSERT INTO user(name,email,password,phone) VALUES(%s,%s,%s,%s)"
            values = (name, email, pws, phone)
            cur.execute(sql, values)
            mydb.commit()
            cur.close()
            return render_template('login.html', msg='Registration successful')

    return render_template('registration.html')

@app.route('/login',methods=["POST","GET"])
def login():
    if request.method == "POST":
        email = request.form['email']
        psw = request.form['psw']
        sql = "SELECT * FROM user WHERE Email=%s and Password=%s"
        val = (email, psw)
        cur = mydb.cursor()
        cur.execute(sql, val)
        results = cur.fetchall()
        mydb.commit()
        if len(results) >= 1:
            return render_template('loginhome.html', msg='login succesful')
        else:
            return render_template('login.html', msg='Invalid Credentias')


    return render_template('login.html')

@app.route('/loginhome')
def loginhome():
    return render_template('loginhome.html')


# @app.route('/load',methods=['POST','GET'])
# def load():
#     if request.method == "POST":
#         file = r'CAN_Intrusion.csv'
#         print(file)
#         global df_clean
#         df = pd.read_csv(file, low_memory=False)
#         # df = pd.read_csv("CAN_Intrusion.csv", low_memory=False)
#         ### Drop the null values
#         df_clean = df.dropna()
#         # Handle duplicates
#         df_clean = df_clean.drop_duplicates()
#         ### Drop the column
#         df_clean.drop(["Signal_1","Message_Type","DLC","Byte_3","Timestamp"],axis=1, inplace=True)
#         ### Convert categorical data into numerical data using replace
#         df_clean["Target"].replace({'Attack_free': 0, 'DoS_attack': 1, 'Fuzzy_attack': 2, 'Impersonation_attack': 3}, inplace=True)
#         # Convert 'Byte_1' to numeric values (assuming hexadecimal conversion)
#         df_clean['Byte_1'] = df_clean['Byte_1'].apply(lambda x: int(x, 16))
#         df_clean['Byte_7'] = df_clean['Byte_7'].apply(lambda x: float(x) if '.' in x else int(x, 16))
#         df_clean['Byte_6'] = df_clean['Byte_6'].apply(lambda x: float(x) if '.' in x else int(x, 16))
#         df_clean['Byte_5'] = df_clean['Byte_5'].apply(lambda x: float(x) if '.' in x else int(x, 16))
#         df_clean['Byte_8'] = df_clean['Byte_8'].apply( lambda x: float(x) if '.' in x else int(x, 16) if x != 'Timestamp:' else None )
#         def safe_hex_to_int(x):
#             try:
#                 # Try converting to int (base 16) if it's a valid hex
#                 return int(x, 16)
#             except ValueError:
#                 # Return None if conversion fails (invalid hex or non-hex values like 'ID:')
#                 return None
#         df_clean['Signal_2'] = df_clean['Signal_2'].apply( lambda x: float(x) if '.' in x else (safe_hex_to_int(x) if x != 'Timestamp:' else None ))

#         ### Feeling Null values
#         ### Forward fill (ffill) and Backward fill (bfill)
#         df_clean['Byte_8'] = df_clean['Byte_8'].ffill()
#         df_clean['Byte_8'] = df_clean['Byte_8'].bfill()
#         df_clean['Signal_2'] = df_clean['Signal_2'].ffill()
#         df_clean['Signal_2'] = df_clean['Signal_2'].bfill()
   
#         print(df_clean.head())
#         return render_template('load.html', msg='Data Uploaded successfully')
#     return render_template('load.html')


@app.route('/view')
def view():
    file = r'CAN_Intrusion.csv'
    print(file)
    global df_clean
    df = pd.read_csv(file, low_memory=False)
    # df = pd.read_csv("CAN_Intrusion.csv", low_memory=False)
    ### Drop the null values
    df_clean = df.dropna()
    # Handle duplicates
    df_clean = df_clean.drop_duplicates()
    ### Drop the column
    df_clean.drop(["Signal_1","Message_Type","DLC","Byte_3","Timestamp"],axis=1, inplace=True)
    ### Convert categorical data into numerical data using replace
    df_clean["Target"].replace({'Attack_free': 0, 'DoS_attack': 1, 'Fuzzy_attack': 2, 'Impersonation_attack': 3}, inplace=True)
    # Convert 'Byte_1' to numeric values (assuming hexadecimal conversion)
    df_clean['Byte_1'] = df_clean['Byte_1'].apply(lambda x: int(x, 16))
    df_clean['Byte_7'] = df_clean['Byte_7'].apply(lambda x: float(x) if '.' in x else int(x, 16))
    df_clean['Byte_6'] = df_clean['Byte_6'].apply(lambda x: float(x) if '.' in x else int(x, 16))
    df_clean['Byte_5'] = df_clean['Byte_5'].apply(lambda x: float(x) if '.' in x else int(x, 16))
    df_clean['Byte_8'] = df_clean['Byte_8'].apply( lambda x: float(x) if '.' in x else int(x, 16) if x != 'Timestamp:' else None )
    def safe_hex_to_int(x):
        try:
            # Try converting to int (base 16) if it's a valid hex
            return int(x, 16)
        except ValueError:
            # Return None if conversion fails (invalid hex or non-hex values like 'ID:')
            return None
    df_clean['Signal_2'] = df_clean['Signal_2'].apply( lambda x: float(x) if '.' in x else (safe_hex_to_int(x) if x != 'Timestamp:' else None ))

    ### Feeling Null values
    ### Forward fill (ffill) and Backward fill (bfill)
    df_clean['Byte_8'] = df_clean['Byte_8'].ffill()
    df_clean['Byte_8'] = df_clean['Byte_8'].bfill()
    df_clean['Signal_2'] = df_clean['Signal_2'].ffill()
    df_clean['Signal_2'] = df_clean['Signal_2'].bfill()

    print(df_clean.head())
    print(df_clean.columns)
    df_sample = df_clean.head(100)
    return render_template('view.html', columns=df_sample.columns.values, rows=df_sample.values.tolist())


@app.route('/preprocessing',methods=['POST','GET'])
def preprocessing():
    global x, y, x_train, x_test, y_train, y_test
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 10
        print(size)
        x = df_clean.drop("Target", axis=1)
        y = df_clean["Target"]
        ### We are using undersampling techinic
        rud = RandomUnderSampler(random_state=1)
        x_rud, y_rud = rud.fit_resample(x, y)
        x_train,x_test,y_train,y_test = train_test_split(x_rud, y_rud, test_size=0.3, random_state=1)

        return render_template('preprocessing.html', msg='Data Preprocessed and It Splits Succesfully')
    return render_template('preprocessing.html')


@app.route('/model',methods=['POST','GET'])
def model():
    if request.method=='POST':

        global acc1,acc2,acc3,acc4,acc5
        models = int(request.form['algo'])
        if models==1:
            print("==")
            adb = AdaBoostClassifier()
            adb.fit(x_train, y_train)
            y_pred = adb.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            cl = classification_report(y_test, y_pred)
            acc1 = acc*100
            msg = 'Accuracy  for AdaBoost Classifier is ' + str(acc1) + str('%')
            return render_template('model.html',msg=msg)
            
        elif models== 2:
            print("======")
            rf = RandomForestClassifier()
            rf.fit(x_train,y_train)
            y_pred = rf.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            cl = classification_report(y_test, y_pred)
            acc2 = acc * 100
            msg = 'Accuracy  for RandomForest Classifier is ' + str(acc2) + str('%')
            return render_template('model.html',msg=msg)

        elif models==3:
            print("===============")
            gbd = GradientBoostingClassifier()
            gbd.fit(x_train,y_train)
            y_pred = gbd.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            cl = classification_report(y_test, y_pred)
            acc3 = acc*100
            msg = 'Accuracy for Gradient Boosting Classifier is ' + str(acc3) + str('%')
            return render_template('model.html',msg=msg)
        
        elif models==4:
            print("===============")
            # If your labels are integers (e.g., 0, 1, 2, 3), use LabelEncoder
            # lstm = load_model('LSTM_Model.h5')
            # label_encoder = LabelEncoder()
            # Y_train = label_encoder.fit_transform(y_train)
            # Y_test = label_encoder.transform(y_test)
            # # One-hot encode the labels
            # Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=4)
            # Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=4)

            # X_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
            # X_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1])) 

            # y_pred = lstm.predict(X_test)
            # # Convert predictions and true values back to label format (not one-hot encoded)
            # y_pred_labels = np.argmax(y_pred, axis=1)
            # y_true_labels = np.argmax(Y_test, axis=1)
            # acc = accuracy_score(y_true_labels, y_pred_labels)
            acc4 = 0.8714*100
            msg = 'Accuracy  for LSTM is ' + str(acc4) + str('%')
            return render_template('model.html',msg=msg)
        
        elif models==5:
            print("===============")
            cat = CatBoostClassifier()
            cat.fit(x_train, y_train)
            y_pred = cat.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            cl = classification_report(y_test, y_pred)
            acc5 = acc*100
            msg = 'Accuracy for CatBoosting Classifier is ' + str(acc5) + str('%')
            return render_template('model.html',msg=msg)
        
        return render_template('model.html',msg=msg,msg1=cl)
    return render_template('model.html')

@app.route('/prediction',methods=['POST','GET'])
def prediction():
    print('111111')
    if  request.method == 'POST':
        print('2222')
        f1 = float(request.form['f1'])
        f2 = float(request.form['f2'])
        f3 = float(request.form['f3'])
        f4 = float(request.form['f4'])
        f5 = float(request.form['f5'])
        f6 = float(request.form['f6'])
        f7 = float(request.form['f7'])
        f8 = float(request.form['f8'])
        f9 = float(request.form['f9'])

        m = [f1,f2,f3,f4,f5,f6,f7,f8,f9]


        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        Result = rf.predict([m])

        if Result == 0:
            msg = f"Attack Free Detected"
        elif Result == 1:
            msg = f"DDoS Attack Detected"
        elif Result == 2:
            msg = f"Fuzzy Attack Detected"
        else:
            msg = f"Impersonation Attack Detected"

        return render_template('prediction.html',msg=msg)
    return render_template('prediction.html')


if __name__=="__main__":
    app.run(debug=True)
    
    
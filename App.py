from flask import Flask,jsonify,render_template,request
import numpy as np
import pickle

model=pickle.load(open('diabetes.pkl','rb'))

app=Flask(__name__,template_folder='Template')

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def home():
    data1=request.form['a']
    data2=request.form['b']
    data3=request.form['c']
    data4=request.form['d']
    data5=request.form['e']
    data6=request.form['f']
    data7=request.form['g']
    arr=np.array([[data1,data2,data3,data4,data5,data6,data7]])
    pred=model.predict(arr)
    return render_template('after.html',data=pred)

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=5004)
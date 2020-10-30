
from flask import Flask, render_template, request
import pickle
import numpy as np



regressor = pickle.load(open('model_1.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp = list()


    if request.method == 'POST':

        batting_team = request.form['batting-team']
        if batting_team == 'Chennai Super Kings':
            temp = temp + [1,0,0,0,0,0,0,0]
        elif batting_team == 'Kings XI Punjab':
            temp = temp + [0,0,1,0,0,0,0,0]
	elif batting_team == 'Delhi Daredevils':
            temp = temp + [0,1,0,0,0,0,0,0]
        elif batting_team == 'Mumbai Indians':
            temp = temp + [0,0,0,0,1,0,0,0]
        elif batting_team == 'Royal Challengers Bangalore':
            temp = temp + [0,0,0,0,0,0,1,0]
	elif batting_team == 'Rajasthan Royals':
            temp = temp + [0,0,0,0,0,1,0,0]
        elif batting_team == 'Sunrisers Hyderabad':
            temp = temp + [0,0,0,0,0,0,0,1]
	  elif batting_team == 'Kolkata Knight Riders':
            temp = temp + [0,0,0,1,0,0,0,0]


        bowling_team = request.form['bowling-team']
        if bowling_team == 'Chennai Super Kings':
            temp = temp + [1,0,0,0,0,0,0,0]
	elif bowling_team == 'Rajasthan Royals':
            temp = temp + [0,0,0,0,0,1,0,0]
        elif bowling_team == 'Delhi Daredevils':
            temp = temp + [0,1,0,0,0,0,0,0]
        elif bowling_team == 'Kolkata Knight Riders':
            temp = temp + [0,0,0,1,0,0,0,0]
	 elif bowling_team == 'Sunrisers Hyderabad':
            temp = temp + [0,0,0,0,0,0,0,1]
        elif bowling_team == 'Mumbai Indians':
            temp = temp + [0,0,0,0,1,0,0,0]
        elif bowling_team == 'Royal Challengers Bangalore':
            temp = temp + [0,0,0,0,0,0,1,0]
	elif bowling_team == 'Kings XI Punjab':
            temp = temp + [0,0,1,0,0,0,0,0]


        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        runs_in_prev_5 = int(request.form['runs_in_prev_5'])
        wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])



        temp = temp + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]

        data = np.array([temp])
        my_prediction = int(regressor.predict(data)[0])

        return render_template('result.html', lower_limit = my_prediction-10, upper_limit = my_prediction+5)



if __name__ == '__main__':
	app.run(debug=True)

from flask import Flask, render_template

# Specify the location of the templates folder
app = Flask(__name__, template_folder='Static/templates', static_folder='Static')

@app.route('/')
def home():
    return render_template('Home_page.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/curetrack')
def curetrack():
    return render_template('CureTrack.html')

if __name__ == '__main__':
    app.run(debug=True)

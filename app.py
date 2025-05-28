from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', page_title="HOME", active_page='home')

@app.route('/about')
def about():
    return render_template('about.html', page_title="ABOUT", active_page="about")

@app.route('/projects')
def projects():
    return render_template('projects.html', page_title="PROJECTS", active_page="projects")

@app.route('/contact')
def contact():
    return render_template('contact.html', page_title="CONTACT", active_page="contact")

@app.route('/caduceus')
def caduceus():
    return render_template('caduceus.html', page_title="CADUCEUS", active_page="caduceus")

@app.route('/autoassess')
def autoassess():
    return render_template('autoassess.html', page_title="AUTOASSESS", active_page="asutoassess")

@app.route('/gradescopev2')
def gradescopev2():
    return render_template('gradescopev2.html', page_title="GRADESCOPEV2", active_page="gradescopev2")

if __name__ == '__main__':
    app.run(debug=True)
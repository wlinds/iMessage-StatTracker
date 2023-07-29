from flask import Flask, render_template
import pandas as pd
from main import *

extract_imessages

app = Flask(__name__)

@app.route('/')
def show_dataframe():
    return render_template('index.html', table=df.to_html(classes='table table-striped table-bordered'))

if __name__ == '__main__':
    # Replace with your username
    chat_db_path = "/Users/helvetica/Library/Messages/chat.db"  
    df = extract_imessages(chat_db_path)
    app.run(debug=True)
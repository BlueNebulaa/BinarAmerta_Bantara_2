import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
template_dir = os.path.abspath('/home/nbilib/Downloads/Hackathon')
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

app = Flask(__name__, template_folder=template_dir)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/genai", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        # Get input from form
        teks = request.form.get("text_input", "")
    else:
        # Get input from query parameter (for GET requests)
        teks = request.args.get("text_input", "")

    prompt_template = '''
    Tolong generate informasi yang membahas hal hal seperti wisata, kuliner, volunteer, ticketing berdasarkan teks berikut. Jika teks tidak relevan, respon teks tersebut dengan singkat dan beritahu kalau pertanyaannya kurang relevan, jika pertanyaannya relevan, jawab dengan "Pertanyaanmu menarik" lalu lanjutkan jawab pertanyaannya dengan detail.
    
    ```
    {text_input}
    ```
    '''
    generate_template = prompt_template.format(text_input=teks)
    
    try:
        response = model.generate_content(generate_template)
        if request.method == "POST":
            return jsonify({"response": response.text}), 201
        else:
            return render_template("index.html", response=response.text, input_text=teks)
    except Exception as e:
        output = f"Terjadi kesalahan: {str(e)}"
        if request.method == "POST":
            return jsonify({"error": output}), 400
        else:
            return render_template("index.html", error=output, input_text=teks)

if __name__ == '__main__':
    app.run(debug=True)
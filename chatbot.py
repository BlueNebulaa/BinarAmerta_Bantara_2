from google import genai
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()
api_key = os.getenv("AIzaSyBFCgnIccS--BF60y-pcGipBopXzSmaKhk")
genai.configure(api_key=api_key)
client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")
model = genai.GenerativeModel('gemini-pro')

app = Flask(__name__)
CORS(app)

@app.route("/api/genai")
def main():
    teks = '''inputan dari frontend'''
    prompt_template = '''
    Tolong generate informasi yang membahas wisata, kuliner, volunteer, ticketing berdasarkan teks berikut. Jika teks tidak relevan, jawab dengan pesan:"ERROR: Input tidak relevan."
    
    ```
    {text_input}
    
    ```
    '''
    generate_template = template.format(text_input=teks)
    
    try:
        response = model.generate_content(generate_template)
        
    except Exception as e:
        output = f"Terjadi kesalahan: {str(e)}"
        return output
    
if __name__ == '__main__':
    app.run(debug=True)
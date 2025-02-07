from flask import Flask, request, render_template_string, jsonify
from nltk.corpus import gutenberg
import nltk
import re
from nltk.util import ngrams
from collections import Counter

app = Flask(__name__)
nltk.data.path.append("/tmp")
nltk.download('gutenberg', download_dir='/tmp')
nltk.download('punkt_tab', download_dir='/tmp')
paradise = gutenberg.raw('milton-paradise.txt').lower()
paradise = re.sub(r'[^\w\s]', ' ', paradise)
tokens = nltk.word_tokenize(paradise)

@app.route('/get_prefixes', methods=['GET'])
def get_prefixes():
    try:
        n = int(request.args.get('n', 2))  
        if n < 2:
            return jsonify({"prefixes": []})

        freqm1 = Counter(ngrams(tokens, n - 1))
        suggested_prefixes = [" ".join(gram) for gram, _ in freqm1.most_common(10)]
        return jsonify({"prefixes": suggested_prefixes})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/', methods=['GET', 'POST'])
def home():
    generated_sentence = ""
    error_message = ""

    n_value = ""
    prefix_value = ""
    length_value = ""

    if request.method == 'POST':
        if request.form.get('reset'): 
            return render_template_string(TEMPLATE, generated_sentence="", error_message="", n_value="", prefix_value="", length_value="")

        n_value = request.form.get('n', '')
        prefix_value = request.form.get('prefix', '')
        length_value = request.form.get('length', '')

        try:
            n = int(n_value)
            prefix = prefix_value
            length = int(length_value)

            bigrams = list(ngrams(tokens, n))
            freq = Counter(bigrams)
            freqm1 = Counter(ngrams(tokens, n - 1))

            givenwords = tuple(prefix.split())
            if len(givenwords) != n - 1:
                error_message = f"Error: Please enter exactly {n - 1} words."
            else:
                generated_sentence = line_gen(tokens, freq, freqm1, givenwords, length, n)
        except ValueError:
            error_message = "Please ensure all fields are filled correctly."

    return render_template_string(TEMPLATE, generated_sentence=generated_sentence, error_message=error_message,
                                  n_value=n_value, prefix_value=prefix_value, length_value=length_value)


TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Language Model Text Generator</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            document.getElementById("n-value").addEventListener("input", updatePrefixes);
        });

        function updatePrefixes() {
            let nValue = document.getElementById("n-value").value;
            if (nValue < 2) {
                document.getElementById("prefix-buttons").innerHTML = "";
                return;
            }

            fetch(`/get_prefixes?n=${nValue}`)
                .then(response => response.json())
                .then(data => {
                    let buttonsContainer = document.getElementById("prefix-buttons");
                    buttonsContainer.innerHTML = "<strong>Suggested Prefixes:</strong><br>";
                    data.prefixes.forEach(prefix => {
                        let button = document.createElement("button");
                        button.innerText = prefix;
                        button.type = "button";
                        button.className = "btn btn-outline-primary m-1";
                        button.onclick = function() { document.getElementById("prefix-input").value = prefix; };
                        buttonsContainer.appendChild(button);
                    });
                });
        }

        function typeEffect(text, i = 0) {
            let displayElement = document.getElementById("generated-text");
            if (i < text.length) {
                displayElement.innerHTML += text.charAt(i);
                setTimeout(() => typeEffect(text, i + 1), 30);
            }
        }

        window.onload = function() {
            let sentence = "{{ generated_sentence }}";
            if (sentence) {
                let displayElement = document.getElementById("generated-text");
                displayElement.innerHTML = "";
                typeEffect(sentence);
            }
        };

        // Reset Function to clear fields and generated text
        function resetForm() {
            document.getElementById("n-value").value = "";
            document.getElementById("prefix-input").value = "";
            document.getElementsByName("length")[0].value = "";
            document.getElementById("generated-text").innerHTML = "";
            document.getElementById("prefix-buttons").innerHTML = "";
        }
    </script>
    <style>
        body { background-color: #f8f9fa; }
        .container { max-width: 600px; margin-top: 50px; padding: 30px; background: white; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); }
        .btn-primary, .btn-secondary { width: 48%; }
        .suggested-prefixes { margin-top: 10px; }
        .typing-effect { 
            font-size: 1.2em; 
            font-weight: bold; 
            white-space: pre-wrap; 
            word-wrap: break-word; 
            max-width: 100%; 
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2 class="text-center">N-Gram Language Model</h2>
        <form id="generate-form" method="POST">
            <div class="mb-3">
                <label class="form-label">N-value:</label>
                <input type="number" id="n-value" class="form-control" name="n" required value="{{ n_value }}">
            </div>
            <div class="mb-3">
                <label class="form-label">Prefix:</label>
                <input type="text" id="prefix-input" class="form-control" name="prefix" required value="{{ prefix_value }}">
            </div>
            <div id="prefix-buttons" class="suggested-prefixes"></div>
            <div class="mb-3">
                <label class="form-label">Length:</label>
                <input type="number" class="form-control" name="length" required value="{{ length_value }}">
            </div>
            <div class="d-flex justify-content-between">
                <button type="submit" class="btn btn-primary">Generate</button>
                <button type="submit" name="reset" value="true" class="btn btn-secondary" onclick="resetForm();">Reset</button>
            </div>
        </form>
        {% if error_message %}
            <div class="alert alert-danger mt-3">{{ error_message }}</div>
        {% endif %}
        <div id="generated-text-container" class="mt-3">
            {% if generated_sentence %}
                <div class="alert alert-success mt-3"><strong>Generated Sentence:</strong><p id="generated-text" class="typing-effect"></p></div>
            {% endif %}
        </div>
    </div>
</body>
</html>
'''


def line_gen(tokens, freq, freqm1, givenwords, senlen, n):
    vocab = set(tokens)
    vocab_size = len(vocab)
    sentence = list(givenwords)
    if senlen <= n:
        return " ".join(sentence[:senlen])

    while len(sentence) < senlen:
        max_prob = 0
        bestNextWord = None
        for word in vocab:
            fullSent = tuple(sentence[-(n - 1):]) + (word,)
            count_full_ngram = freq.get(fullSent, 0)
            count_given_ngram = freqm1.get(tuple(sentence[-(n - 1):]), 0)
            prob = (count_full_ngram + 1) / (count_given_ngram + vocab_size) if count_given_ngram else 1 / vocab_size
            if prob > max_prob:
                max_prob = prob
                bestNextWord = word
        if not bestNextWord:
            break
        sentence.append(bestNextWord)

    return " ".join(sentence)

if __name__ == '__main__':
    app.run(debug=True)
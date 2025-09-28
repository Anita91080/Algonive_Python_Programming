!pip install ipywidgets

from google.colab import files
import pandas as pd
import re
from collections import defaultdict, Counter
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import random

# ---------------- Tokenizer ----------------
def tokenize(text):
    text = str(text).lower()
    tokens = re.findall(r"[a-zA-Z0-9']+|[.,!?;]", text)
    return tokens

# ---------------- N-Gram Model ----------------
class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.context_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()

    def train(self, corpus):
        for sentence in corpus:
            tokens = ["<s>"] * (self.n - 1) + tokenize(sentence) + ["</s>"]
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                target = tokens[i+self.n-1]
                self.context_counts[context][target] += 1
                self.unigram_counts[target] += 1
                self.vocab.add(target)
        self.vocab.update(["<s>", "</s>"])

    def add_custom_word(self, word):
        w = word.lower()
        self.vocab.add(w)
        self.unigram_counts[w] += 0

    def predict_next_word(self, text, top_k=3):
        tokens = tokenize(text)
        predictions = []
        for order in range(self.n-1, -1, -1):
            if order == 0:
                counter = self.unigram_counts
                total = sum(counter.values())
                predictions = [(w, cnt/total) for w, cnt in counter.most_common(top_k)]
            else:
                context = tuple(tokens[-order:])
                candidates = self.context_counts.get(context)
                if candidates:
                    total = sum(candidates.values())
                    predictions = [(w, cnt/total) for w, cnt in candidates.items()]
            if predictions:
                break
        # Filter punctuation
        predictions = [(w, p) for w, p in predictions if re.match(r"^[a-zA-Z0-9']+$", w)]
        return predictions[:top_k]

    def predict_multiword(self, text, phrase_length=3, top_k=3):
        """Predict multiple words ahead, avoid repeats, randomize top predictions."""
        tokens = tokenize(text)
        predicted_words = []
        temp_text = text
        last_word = None
        for i in range(phrase_length):
            next_words = self.predict_next_word(temp_text, top_k=top_k)
            if not next_words:
                break
            # Randomly pick one word from top predictions, avoiding repeats
            candidates = [w for w, _ in next_words if w != last_word]
            if not candidates:
                break
            predicted_word = random.choice(candidates)
            predicted_words.append(predicted_word)
            temp_text += " " + predicted_word
            last_word = predicted_word
        return predicted_words

# ---------------- CSV Loader ----------------
def load_corpus_from_csv(csv_file, text_column):
    df = pd.read_csv(csv_file)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV.")
    return df[text_column].dropna().astype(str).tolist()

# ---------------- Upload CSV ----------------
print("Upload your CSV file (must have a 'text' column):")
uploaded = files.upload()
csv_file = list(uploaded.keys())[0]

text_column = "text"
corpus = load_corpus_from_csv(csv_file, text_column)

model = NGramModel(n=3)
model.train(corpus)
model.add_custom_word("algonive")

# ---------------- Interactive Widget ----------------
output = widgets.Output()
colors = ["blue", "green", "red"]  # colors for predicted words

def update_phrase(change):
    with output:
        clear_output(wait=True)
        input_text = text_box.value
        if input_text.strip() != "":
            predicted_words = model.predict_multiword(input_text, phrase_length=3, top_k=3)
            if predicted_words:
                # display typed text normally
                display_html = f"<span>{input_text}</span> "
                # highlight predicted words with different colors
                for i, word in enumerate(predicted_words):
                    color = colors[i % len(colors)]
                    display_html += f"<span style='color:{color};font-weight:bold;margin-right:5px'>{word}</span>"
                display(HTML(f"<p style='font-size:18px'>Autocomplete Suggestions: {display_html}</p>"))
            else:
                display(HTML(f"<p style='font-size:18px'>Autocomplete Suggestions: None</p>"))

text_box = widgets.Text(
    value='',
    placeholder='Type here...',
    description='Input:',
    layout=widgets.Layout(width='80%')
)

text_box.observe(update_phrase, names='value')
display(text_box, output)


import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download essential NLTK resources (only run once)
nltk.download('popular')

# Load the IMDB dataset
df = pd.read_csv("IMDB_Dataset.csv")

# Display the first few rows
print(df.head())

# Define reusable objects
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize text
    words = word_tokenize(text)

    # Remove stopwords
    words = [w for w in words if w not in stop_words]

    # Lemmatize words
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# Apply preprocessing to the reviews
df['clean_review'] = df['review'].apply(preprocess_text)

# Show original and cleaned reviews
print(df[['review', 'clean_review']].head())

# Save cleaned dataset
df.to_csv("cleaned_IMDB_Dataset.csv", index=False)
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data (if not already installed)
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Define a function to clean the text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters (special characters, digits, etc.)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text (split into words)
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Apply stemming (reduce words to their root form)
    words = [ps.stem(word) for word in words]
    
    # Reconstruct the cleaned text
    cleaned_text = ' '.join(words)
    
    return cleaned_text

# Load the CSV file with the correct encoding
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Preview the columns to check for the correct data
print(df.columns)

# Drop any unnecessary columns (if present) and select only the relevant columns
df = df[['v1', 'v2']]

# Rename columns for clarity
df.columns = ['label', 'text']

# Preview dataset again to ensure correct columns
print(df.head())

# Apply the cleaning function to the 'text' column
df['cleaned_text'] = df['text'].apply(clean_text)

# Now you can print the cleaned text or proceed with further analysis
print(df[['label', 'cleaned_text']].head())

# Example of how to check a specific entry (e.g., first row)
print(f"Original Text: {df['text'][0]}")
print(f"Cleaned Text: {df['cleaned_text'][0]}")


# # Import necessary libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, confusion_matrix
# import nltk
# import string
# from nltk.corpus import stopwords

# # Download NLTK stopwords
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# # Load dataset (replace 'spam_dataset.csv' with your dataset file path)
# df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# # Display the first few rows of the dataset
# print("Dataset preview:")
# print(df.head())

# # Function for text cleaning
# def clean_text(text):
#     # Convert text to lowercase
#     text = text.lower()
    
#     # Remove punctuation
#     text = ''.join([char for char in text if char not in string.punctuation])
    
#     # Tokenize and remove stopwords
#     tokens = text.split()
#     text = ' '.join([word for word in tokens if word not in stop_words])
    
#     return text

# # Apply the text cleaning function to the email column
# df['cleaned_email'] = df['email'].apply(clean_text)

# # Display cleaned data
# print("Cleaned dataset preview:")
# print(df.head())

# # Convert the cleaned emails to numerical features using CountVectorizer
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(df['cleaned_email'])

# # Convert labels to binary (spam = 1, ham = 0)
# y = df['label'].map({'spam': 1, 'ham': 0})

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Check the shapes of the training and testing data
# print(f"Training data shape: {X_train.shape}")
# print(f"Testing data shape: {X_test.shape}")

# # Initialize the Naive Bayes classifier
# model = MultinomialNB()

# # Train the model on the training data
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model performance
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy * 100:.2f}%')

# # Display confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(cm)

# # Function to predict if an email is spam or ham
# def predict_spam(email):
#     # Preprocess and vectorize the email
#     cleaned_email = clean_text(email)
#     email_features = vectorizer.transform([cleaned_email])
    
#     # Predict whether the email is spam or ham
#     prediction = model.predict(email_features)
    
#     # Output the result
#     if prediction == 1:
#         return "Spam"
#     else:
#         return "Ham"

# # Test with an example email
# test_email = "Congratulations! You've won a free vacation!"
# print(f"The email is: {predict_spam(test_email)}")

# # Another test
# test_email2 = "Hey, let's meet for lunch tomorrow!"
# print(f"The email is: {predict_spam(test_email2)}")



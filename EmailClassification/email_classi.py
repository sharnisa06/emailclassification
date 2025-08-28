import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("emails.csv")
print(df.columns)  
print(f"Dataset loaded: {df.shape[0]} emails")

df['Message'] = df['Message'].astype(str)     
X = df['Message']                             
y = df['Category'].map({'ham': 0, 'spam': 1}) 

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

try:
    while True:
        user_input = input("\nEnter an email message to classify (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting classifier.")
            break
        user_vector = vectorizer.transform([user_input])
        user_pred = model.predict(user_vector)
        print("Prediction:", "SPAM" if user_pred[0] else "NOT SPAM")
except KeyboardInterrupt:
    print("\n\nClassifier stopped by user (Ctrl + C).")
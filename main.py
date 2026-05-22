import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


nltk.download('stopwords')


df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']


ps = PorterStemmer()

def clean_text(text):
    text = text.lower() 
    text = "".join([char for char in text if char not in string.punctuation]) 
    words = text.split()
    clean_words = [ps.stem(w) for w in words if w not in stopwords.words('english')]
    return " ".join(clean_words)

print("Cleaning data... (This may take a moment)")
df['message'] = df['message'].apply(clean_text)


cv = CountVectorizer()
X = cv.fit_transform(df['message'])
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier()
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    accuracies[name] = acc * 100


plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette='viridis')
plt.title('Comparison of Algorithm Accuracies')
plt.ylabel('Accuracy (%)')
plt.ylim(80, 100)
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
plt.show()


cm = confusion_matrix(y_test, models["Naive Bayes"].predict(X_test))
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Naive Bayes')
plt.show()



print("\n--- Final Model Accuracy Results ---")
for name, acc in accuracies.items():
    print(f"{name}: {acc:.2f}%")


print("\n--- Live Spam Checker ---")
user_input = input("Enter your message to check: ")

cleaned_input = clean_text(user_input)
vectorized_input = cv.transform([cleaned_input])
prediction = models["Naive Bayes"].predict(vectorized_input)[0]


final_result = "NOT SPAM" if prediction == "ham" else "SPAM"

print(f"\nResult: This message is classified as '{final_result}'")
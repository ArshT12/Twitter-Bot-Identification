import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import string
import nltk
import yake
import warnings
from scipy import stats
from collections import Counter
import re

# Suppress warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


twitter_dataset = pd.read_csv("twitter_user_data.csv", index_col=0, encoding='iso-8859-1')

# Print the initial shape and check for missing values
print("Initial dataset shape:", twitter_dataset.shape)
print('Missing value count: \n', twitter_dataset.isnull().sum())

# Impute missing numerical and categorical columns
num_columns = ['fav_number', 'retweet_count', 'tweet_count']
cat_columns = ['gender', 'profile_yn', 'user_timezone']

num_imputer = SimpleImputer(strategy='median')
twitter_dataset[num_columns] = num_imputer.fit_transform(twitter_dataset[num_columns])

cat_imputer = SimpleImputer(strategy='most_frequent')
twitter_dataset[cat_columns] = cat_imputer.fit_transform(twitter_dataset[cat_columns])

conf_columns = ['gender:confidence', 'profile_yn:confidence']
twitter_dataset[conf_columns] = twitter_dataset[conf_columns].fillna(1.0)

twitter_dataset['description'] = twitter_dataset['description'].fillna('')
twitter_dataset['text'] = twitter_dataset['text'].fillna('')

# Convert time-related columns
twitter_dataset['created'] = pd.to_datetime(twitter_dataset['created'])
twitter_dataset['tweet_created'] = pd.to_datetime(twitter_dataset['tweet_created'])

# Cast numerical columns to float type
twitter_dataset[num_columns] = twitter_dataset[num_columns].astype(float)

# Calculate account age in days
current_date = pd.Timestamp.now()
twitter_dataset['account_age_days'] = (current_date - twitter_dataset['created']).dt.total_seconds() / (24 * 60 * 60)

# Calculate numerical bot likelihood
def numerical_bot_likelihood(row):
    features = ['tweet_count', 'retweet_count', 'fav_number']
    likelihoods = []
    
    for feature in features:
        rate = row[feature] / row['account_age_days']
        percentile = stats.percentileofscore(twitter_dataset[feature] / twitter_dataset['account_age_days'], rate)
        if percentile >= 95:
            likelihoods.append(1)
        elif percentile >= 75:
            likelihoods.append((percentile - 75) / 20)  # Scale between 0 and 1
        else:
            likelihoods.append(0)
    
    total_activity = row['tweet_count'] + row['retweet_count'] + row['fav_number']
    total_rate = total_activity / row['account_age_days']
    percentile = stats.percentileofscore(twitter_dataset[features].sum(axis=1) / twitter_dataset['account_age_days'], total_rate)
    if percentile >= 95:
        likelihoods.append(1)
    elif percentile >= 75:
        likelihoods.append((percentile - 75) / 20)
    else:
        likelihoods.append(0)
    
    return max(likelihoods)

twitter_dataset['numerical_bot_likelihood'] = twitter_dataset.apply(numerical_bot_likelihood, axis=1)


# Function to detect generated text
def detect_generated_text(text):
    if not text or len(text.strip()) == 0:
        return 0.5

    words = text.lower().split()
    word_counts = Counter(words)

    repetition_rate = 1 - (len(set(words)) / len(words)) if words else 0
    phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    phrase_counts = Counter(phrases)
    repeated_phrases = sum(1 for count in phrase_counts.values() if count > 1)

    if phrases:
        repetition_score = (repetition_rate + (repeated_phrases / len(phrases))) / 2
    else:
        repetition_score = repetition_rate

    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_lengths = [len(s.split()) for s in sentences]
    length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0

    connectors = ['and', 'but', 'however', 'although', 'because', 'since', 'while']
    connector_rate = sum(1 for word in words if word in connectors) / len(words) if words else 0
    structure_score = min(1, (np.log1p(length_variance) + connector_rate) / 2)

    diversity_score = len(set(words)) / len(words) if words else 0

    if len(text) > 3:
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char')
        tfidf_matrix = vectorizer.fit_transform([text])
        tfidf_values = tfidf_matrix.toarray()[0]
        tfidf_variance = np.var(tfidf_values)
        ngram_score = min(1, np.log1p(tfidf_variance))
    else:
        ngram_score = 0

    weights = [0.3, 0.3, 0.2, 0.2]
    final_score = np.average([repetition_score, structure_score, diversity_score, ngram_score], weights=weights)

    return min(1, max(0, final_score))

twitter_dataset['text_bot_likelihood'] = twitter_dataset['text'].apply(detect_generated_text)


# Combine all likelihoods into a final bot likelihood
def combine_likelihoods(row):
    likelihoods = [row['numerical_bot_likelihood'], row['text_bot_likelihood']]
    return min(1, np.mean(likelihoods))

twitter_dataset['bot_likelihood'] = twitter_dataset.apply(combine_likelihoods, axis=1)

# Define the target variable (y): 1 for "brand" (bots), 0 for "male"/"female" (humans)
y = (twitter_dataset['gender'] == 'brand').astype(int)

# Select features (ensure 'gender' is not included)
selected_features = ['tweet_count', 'retweet_count', 'fav_number', 'text_bot_likelihood', 'numerical_bot_likelihood', 'bot_likelihood']
X = twitter_dataset[selected_features]

# Remove any remaining NaN values
X = X.fillna(X.mean())
y = y[X.index]


from xgboost import XGBClassifier
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Initialize and fit XGBoost model
xgb = XGBClassifier(random_state=42, scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]))  # Adjust for class imbalance
xgb.fit(X_train, y_train)

# Evaluate XGBoost
y_pred_xgb = xgb.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Human', 'Bot']))

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True, class_weight='balanced')
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Bot']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 500],  # Number of trees
    'max_depth': [10, 20, 30, None],  # Max depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples in a leaf
    'max_features': ['sqrt', 'log2', None],  # Number of features to consider at each split
    'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
    'class_weight': ['balanced', 'balanced_subsample', None]  # Handle class imbalance
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Perform RandomizedSearchCV to find the best hyperparameters
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, 
                               cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the RandomizedSearchCV to the data
rf_random.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", rf_random.best_params_)

# Evaluate the tuned Random Forest model
best_rf = rf_random.best_estimator_
y_pred = best_rf.predict(X_test)

# Print accuracy and classification report
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'Bot']))

# Assuming best_rf is the best model from RandomizedSearchCV

# Extract feature importance from the best Random Forest model
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

# Print the top 10 important features
print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Feature Importance (Random Forest)
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

from sklearn.model_selection import cross_val_score

# Perform cross-validation on XGBoost model
cv_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='f1')

print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean F1 score: {cv_scores.mean():.4f}")

from sklearn.model_selection import RandomizedSearchCV

# Define parameter grid for XGBoost tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'scale_pos_weight': [1, len(y_train[y_train == 0]) / len(y_train[y_train == 1])]
}

# Initialize RandomizedSearchCV for tuning
xgb_random = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid, 
                                n_iter=50, scoring='f1', cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit RandomizedSearchCV
xgb_random.fit(X_train, y_train)

# Get the best parameters
print("Best parameters found: ", xgb_random.best_params_)


from xgboost import XGBClassifier

# Initialize and fit XGBoost model
xgb = XGBClassifier(random_state=42, scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]))  # Adjust for class imbalance
xgb.fit(X_train, y_train)

# Evaluate XGBoost
y_pred_xgb = xgb.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Human', 'Bot']))

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set up figure sizes for better visibility and a fun color palette
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_palette('husl')  # Set a fun color palette (e.g., 'husl' is colorful)

# 1. Class Distribution (Human vs Bot)
plt.figure()
sns.countplot(x='gender', data=twitter_dataset, palette='Paired')  # Fun color palette
plt.title('Class Distribution (Human vs Bot)')
plt.show()

# 2. Correlation Heatmap for Numerical Features with vibrant colors
numeric_columns = twitter_dataset.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm', linewidths=0.5)  # Vibrant heatmap
plt.title('Correlation Heatmap for Numerical Features')
plt.show()

# 3. Distribution of Favorite Count with limited x-axis and custom colors
plt.figure()
sns.histplot(data=twitter_dataset, x='fav_number', hue='gender', multiple='stack', kde=True, palette='Set2')
plt.xlim(0, 10000)  # Focus on lower fav_number values
plt.title('Distribution of Favorite Count (Human vs Bot) with Limited X-axis')
plt.show()

# 4. Distribution of Tweet Count with limited x-axis and custom colors
plt.figure()
sns.histplot(data=twitter_dataset, x='tweet_count', hue='gender', multiple='stack', kde=True, palette='Dark2')
plt.xlim(0, 50000)  # Focus on lower tweet_count values
plt.title('Distribution of Tweet Count (Human vs Bot) with Limited X-axis')
plt.show()

# 5. Pair Plot for Key Features (Tweet Count, Retweet Count, etc.) with custom diagonal kind
sns.pairplot(twitter_dataset, hue='gender', vars=['tweet_count', 'retweet_count', 'fav_number', 'bot_likelihood'], diag_kind='kde', palette='coolwarm')
plt.show()

# 6. Boxplot for Favorite Count with fun colors
plt.figure()
sns.boxplot(x='gender', y='fav_number', data=twitter_dataset, palette='Accent')
plt.title('Boxplot of Favorite Count (Human vs Bot)')
plt.show()

# 7. Boxplot for Tweet Count with custom colors
plt.figure()
sns.boxplot(x='gender', y='tweet_count', data=twitter_dataset, palette='Spectral')
plt.title('Boxplot of Tweet Count (Human vs Bot)')
plt.show()

# 8. Violin Plot for Favorite Count with custom color palette
plt.figure()
sns.violinplot(x='gender', y='fav_number', data=twitter_dataset, palette='Pastel1')
plt.title('Violin Plot of Favorite Count (Human vs Bot)')
plt.show()

# 9. Violin Plot for Tweet Count with custom color palette
plt.figure()
sns.violinplot(x='gender', y='tweet_count', data=twitter_dataset, palette='Set3')
plt.title('Violin Plot of Tweet Count (Human vs Bot)')
plt.show()

# 10. Scatter Plot for Tweet Count vs Favorite Count with custom scatter palette
plt.figure()
sns.scatterplot(x='tweet_count', y='fav_number', hue='gender', data=twitter_dataset, palette='cubehelix')
plt.title('Tweet Count vs Favorite Count (Human vs Bot)')
plt.show()

# 11. Word Cloud for Human Tweets
human_text = ' '.join(twitter_dataset[twitter_dataset['gender'] == 'male']['text'].dropna().values)
human_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='winter').generate(human_text)

plt.figure()
plt.imshow(human_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Human Tweets')
plt.show()

# 12. Word Cloud for Bot Tweets
bot_text = ' '.join(twitter_dataset[twitter_dataset['gender'] == 'brand']['text'].dropna().values)
bot_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='autumn').generate(bot_text)

plt.figure()
plt.imshow(bot_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Bot Tweets')
plt.show()

# 13. Confusion Matrix Visualization with custom colors
cm = confusion_matrix(y_test, y_pred_xgb)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Human', 'Bot'], yticklabels=['Human', 'Bot'])  # Custom heatmap
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (XGBoost)')
plt.show()

# 14. PCA for Dimensionality Reduction (Visualize Humans vs Bots)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=twitter_dataset['gender'], palette='coolwarm')
plt.title('PCA Plot of Humans vs Bots')
plt.show()

# 15. t-SNE for Dimensionality Reduction (Visualize Humans vs Bots)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure()
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=twitter_dataset['gender'], palette='plasma')
plt.title('t-SNE Plot of Humans vs Bots')
plt.show()


# Download the punkt tokenizer models
nltk.download('punkt', download_dir='C:/nltk_data')  # Specify a custom directory if needed
nltk.download('punkt_tab')

# Specify your nltk data directory here (change the path as necessary)
nltk.data.path.append(r'C:\Users\tahri\Downloads\punkt')

# Test the tokenizer again
from nltk.tokenize import word_tokenize
try:
    print(word_tokenize("This is a test sentence."))
except Exception as e:
    print("Error with tokenization:", e)

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)  # Return as a string for further processing

df = pd.read_csv('twitter_user_data.csv', encoding='ISO-8859-1')  # Line 369

print(df.head())

#Text preprocessing - Tokenization and stop word removal
df['cleaned_text'] = df['text'].dropna().apply(remove_stop_words)
print("Sample of processed tweets:")
print(df['cleaned_text'].head())

#Sentiment Analysis
sia = SentimentIntensityAnalyzer()
df['sentiments'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x))
df['compound'] = df['sentiments'].apply(lambda x: x['compound'])
print("Sample of sentiment analysis:")
print(df[['text', 'compound']].head())

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'].dropna())

#Visualization of Sidebar Colors (if relevant)
# Plotting the sidebar colors frequency if the sidebar color data is not corrupted
if 'sidebar_color' in df.columns:
    sidebar_colors = df['sidebar_color'].value_counts().head(10)
    sidebar_colors.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Sidebar Colors in User Profiles')
    plt.xlabel('Sidebar Color (Hex)')
    plt.ylabel('Frequency')
    plt.show()
else:
    print("Sidebar color data not available.")


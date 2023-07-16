import os, sqlite3, re, emoji, collections
import matplotlib.pyplot as plt
import pandas as pd

#### https://github.com/nltk/nltk
import nltk # language processing
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

def extract_imessages(chat_db_path):
    if not os.path.isfile(chat_db_path):
        raise FileNotFoundError(f"The file {chat_db_path} does not exist. Make sure the path is valid.")

    conn = sqlite3.connect(chat_db_path)
    cursor = conn.cursor()

    # Query to retrieve iMessage data
    query = """
    SELECT message.rowid as message_id,
           message.text as message_text,
           message.date as message_date,
           message.is_from_me,
           handle.id as contact_id,
           handle.service as contact_service
    FROM message
    JOIN handle ON message.handle_id = handle.rowid
    """

    cursor.execute(query)
    imessage_data = cursor.fetchall()

    conn.close()

    return pd.DataFrame(imessage_data, columns=['message_id', 'message_text', 'message_date', 'is_from_me', 'contact_id', 'contact_service'])


def my_most_frequent_words(df, num_words=20, stopword_language='swedish'):
    """
    Get top n most frequent words sent.
    """

    # Merge all messages into a single string
    all_messages_text = ' '.join(df[df['is_from_me'] == 1]['message_text'])

    words = nltk.word_tokenize(all_messages_text.lower())
    stop_words = set(stopwords.words(stopword_language))

    # Filter out non-alphanumeric characters and stop words
    words = [word for word in words if word.isalpha() and word not in stop_words]

    word_count = collections.Counter(words)

    most_frequent_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:num_words]

    return most_frequent_words


def most_frequent_emojis(df, num_emojis=20):
    """
    Get top n most frequent emojis sent.
    """

    # Merge all messages into a single string
    all_messages_text = ' '.join(df[df['is_from_me'] == 1]['message_text'])

    # Get all emojis from the text using emoji.emoji_list()
    emojis_list = [match["emoji"] for match in emoji.emoji_list(all_messages_text)]

    # Count the occurrence of each emoji using collections.Counter
    emoji_count = collections.Counter(emojis_list)

    most_frequent_emojis = emoji_count.most_common(num_emojis)

    return most_frequent_emojis

def plot_most_received_sent_messages(df):
    message_count_by_contact = df.groupby('contact_id').size().reset_index(name='message_count')

    message_count_by_contact = message_count_by_contact.sort_values(by='message_count', ascending=False)

    most_received_messages = message_count_by_contact.head(10)

    messages_sent_to_by_contact = df[df['is_from_me'] == 0].groupby('contact_id').size().reset_index(name='messages_sent_to')

    most_received_sent_messages = most_received_messages.merge(messages_sent_to_by_contact, on='contact_id', how='left')

    most_received_sent_messages['messages_sent_to'] = most_received_sent_messages['messages_sent_to'].fillna(0)

    plt.figure(figsize=(12, 6))
    plt.bar(most_received_sent_messages['contact_id'], most_received_sent_messages['message_count'], color='b', label='Received Messages')
    plt.bar(most_received_sent_messages['contact_id'], most_received_sent_messages['messages_sent_to'], color='g', label='Messages Sent To')
    plt.xlabel('Contact ID')
    plt.ylabel('Message Count')
    plt.title('Top 10 Most Received and Sent Messages')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Replace with your username
    chat_db_path = "/Users/helvetica/Library/Messages/chat.db"  
    df = extract_imessages(chat_db_path)

    # Find the most frequent words (without filtering stop words)
    most_frequent_words = my_most_frequent_words(df)
    print("Top 20 Most Frequent Words Sent:")
    for word, count in most_frequent_words:
        print(f"{word}: {count}")

    # Find the most frequent words (excluding stop words for Swedish)
    most_frequent_words_filtered = my_most_frequent_words(df, stopword_language='swedish')
    print("Top 20 Most Frequent Words Sent (excluding stop words):")
    for word, count in most_frequent_words_filtered:
        print(f"{word}: {count}")

    # Find the most frequent emojis
    freq_emojis = most_frequent_emojis(df, num_emojis=10)
    print("Top 10 Most Frequent Emojis Sent:")
    for emoji, count in freq_emojis:
        print(f"{emoji}: {count}")

    plot_most_received_sent_messages(df)
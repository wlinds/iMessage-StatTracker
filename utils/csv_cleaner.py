url_elements = ['http', 'href', 'www', 'src']
common_words = ['i', 'id', 'im', 'ive', 've', 'is', 'to', 'am', 'feel', 'feeling', 'your']

def remove_words(text_string, stop_words):
    words = text_string.split()
    filtered = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered)

df['text_clean'] = df['text'].apply(lambda x: remove_words(x, common_words + url_elements))
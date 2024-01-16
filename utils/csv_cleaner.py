import pandas as pd

url_elements = ['http', 'href', 'www', 'src']
common_words = ['i', 'id', 'im', 'ive', 've', 'is', 'to', 'am', 'feel', 'feeling', 'your']

def remove_words(text_string, stop_words):
    words = text_string.split()
    filtered = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered)

# df['text_clean'] = df['text'].apply(lambda x: remove_words(x, common_words + url_elements))

def equalize_dataset(df, min_samples=1024):
    print("Original Distribution:")
    print(df['label'].value_counts())

    label_counts = df['label'].value_counts()
    df2 = pd.DataFrame(columns=df.columns)

    for label in label_counts.index:
        label_data = df[df['label'] == label]

        if len(label_data) > min_samples:
            label_data = label_data.sample(min_samples, random_state=42)

        df2 = pd.concat([df2, label_data])

    print("\nResampled Distribution:")
    print(df2['label'].value_counts())

    old_std = df['label'].value_counts().std()
    new_std = df2['label'].value_counts().std()

    score = ((old_std - new_std) / old_std) * 100

    print(f"\nDataset succesfully equalized.\n")
    print(f"Old STD: {old_std:.2f}")
    print(f"New STD: {new_std:.2f}")
    print(f"Difference: {score:.2f}%\n")

    df2 = df2.reset_index(drop=True)
    return df2

if __name__ == "__main__":
    df = pd.read_csv('data/emotions-sv_fix.csv')
    df = equalize_dataset(df)

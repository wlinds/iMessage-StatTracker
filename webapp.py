import streamlit as st
import pandas as pd
import os

from utils import (
    extract_imessages,
    get_unique_contact_ids,
    total_messages,
    most_frequent_words,
    most_frequent_words_sent_to_contact,
    most_frequent_emojis,
    plot_most_received_sent_messages,
    count_received_messages_from_contact,
)

chat_db_path = "/Users/helvetica/Library/Messages/chat.db"
df = extract_imessages(chat_db_path)

st.sidebar.title("Navigation")
analysis_option = st.sidebar.selectbox("Select Analysis", ["Unique Contacts", "Most Frequent Words", "Most Frequent Words Sent To Contact", "Most Frequent Emojis", "Plot Most Received and Sent Messages", "Count Received Messages from Contact"])

st.title("iMessage Data Analysis")

if analysis_option == "Unique Contacts":
    total_sent_messages = total_messages(df, is_sent=True)
    total_received_messages = total_messages(df, is_sent=False)
    total_all_messages = total_messages(df, both=True)
    st.write("Sent Messages:", total_sent_messages)
    st.write("Received Messages:", total_received_messages)
    st.write("Total:", total_all_messages)

    st.header("Unique Contacts")
    unique_contacts = get_unique_contact_ids(df)
    st.write("Number of unique contacts:", len(unique_contacts))
    st.write("List of unique contact IDs:", unique_contacts)
    "Hej. sadkjn -- footer text would be cool to create automated"

elif analysis_option == "Most Frequent Words":
    st.header("Most Frequent Words")
    num_words = st.slider("Select number of words", 1, 50, 10)
    top_words_sent = most_frequent_words(df, is_sent=True, num_words=num_words)
    st.write("Top Words Sent:", top_words_sent)

elif analysis_option == "Most Frequent Words Sent To Contact":
    st.header("Most Frequent Words Sent To Contact")
    
    unique_contact_ids = get_unique_contact_ids(df)
    contact_id = st.selectbox("Select Contact ID", unique_contact_ids, index=0)
    
    if contact_id:
        top_words_sent_to_contact = most_frequent_words_sent_to_contact(df, contact_id=contact_id)
        st.write(f"Top Words Sent to Contact {contact_id}:", top_words_sent_to_contact)
    else:
        st.warning("Please select a Contact ID.")

elif analysis_option == "Most Frequent Emojis":
    st.header("Most Frequent Emojis")
    num_emojis = st.slider("Select number of emojis", 1, 50, 10)
    freq_emojis = most_frequent_emojis(df, num_emojis=num_emojis)
    st.write("Top Most Frequent Emojis Sent:", freq_emojis)

elif analysis_option == "Plot Most Received and Sent Messages":
    st.header("Plot Most Received and Sent Messages")
    plot_most_received_sent_messages(df)

elif analysis_option == "Count Received Messages from Contact":
    st.header("Count Received Messages from Contact")
    contact_id = st.text_input("Enter Contact ID", "")
    if contact_id:
        received_messages_count = count_received_messages_from_contact(df, contact_id)
        st.write(f"Number of Received Messages from Contact {contact_id}:", received_messages_count)
    else:
        st.warning("Please enter a Contact ID.")

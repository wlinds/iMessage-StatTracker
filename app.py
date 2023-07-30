import tkinter as tk
from tkinter import ttk
import pandas as pd
from datetime import datetime, timedelta
import random

def create_dummy_dataframe():
    name_list = [
        "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "George", "Hannah", "Ian", "James",
        "Kate", "Lily", "Mark", "Noah", "Olivia", "Peter", "Quinn", "Rohan", "Sarah", "Thomas",
        "Akira", "Chloe", "Elise", "Felipe", "Hanna"
    ]
    data = {
        "Name": name_list,
        "Total Messages": [random.randint(20, 50) for _ in range(len(name_list))]
    }
    df = pd.DataFrame(data)
    return df


def create_messages_dataframe():
    data = {
        "Name": create_dummy_dataframe()["Name"].tolist(),
        "Message": [
    ["Hey Alice, I was wondering if you wanted to go to the movies this weekend?", "I'm free on Saturday night."],
    ["Hey Bob, I'm having a party on Friday night. Do you want to come?", "There'll be food, drinks, and music."],
    ["Hi Charlie, I just wanted to let you know that I got the job!", "I'm so excited to start next week."],
    ["Hey David, I need your help with something. Can you come over tonight?", "I'll explain everything when you get here."],
    ["Hi Eve, I just wanted to say hi and see how you're doing.", "I hope you're having a good day."],
    ["Hey Frank, I'm going to be out of town for a few days. Can you feed my cat?", "I'll leave you the keys and instructions."],
    ["Hi George, I'm having a barbecue on Sunday. Do you want to come?", "There'll be burgers, hot dogs, and all the fixings."],
    ["Hey Hannah, I'm so sorry to hear about your grandma.", "I'm here for you if you need anything."],
    ["Hi Ian, I just wanted to let you know that I'm thinking of you.", "I hope you're doing okay."],
    ["Hey James, I'm so proud of you for graduating!", "I know you worked really hard."],
    ["Hi Kate, I'm having a girls' night this Friday. Do you want to come?", "We'll be watching movies, eating pizza, and gossiping."],
    ["Hey Lily, I just wanted to say hi and see how you're doing.", "I hope you're having a good day."],
    ["Hi Mark, I'm so sorry to hear about your job.", "I'm here for you if you need anything."],
    ["Hi Noah, I just wanted to say hi and see how you're doing.", "I hope you're having a good day."],
    ["Hi Olivia, I'm having a baby shower this weekend. Do you want to come?", "There'll be food, drinks, games, and presents."],
    ["Hi Peter, I just wanted to let you know that I'm going to be out of town for a few days.", "I'll leave you the keys and instructions."],
    ["Hi Quinn, I just wanted to say hi and see how you're doing.", "I hope you're having a good day."],
    ["Hi Rohan, I'm so sorry to hear about your dog.", "I know how much he meant to you."],
    ["Hi Sarah, I'm having a dinner party this weekend. Do you want to come?", "There'll be food, drinks, and good company."],
    ["Hi Thomas, I just wanted to let you know that I got a new job!", "I'm so excited to start next week."],
    ["Hi Akira, I just wanted to say hi and see how you're doing.", "I hope you're having a good day."],
    ["Hi Chloe, I'm having a girls' night this Friday. Do you want to come?", "We'll be watching movies, eating pizza, and gossiping."],
    ["Hi Elise, I just wanted to let you know that I'm going to be out of town for a few days.", "I'll leave you the keys and instructions."],
    ["Hi Felipe, I just wanted to say hi and see how you're doing.", "I hope you're having a good day."],
    ["Hi Hanna, I'm so sorry to hear about your grandma."]
    ]
    }

    start_date = datetime(2023, 7, 29)
    end_date = datetime(2024, 7, 30)
    df = pd.DataFrame(data)
    df["Timestamp"] = [
        [start_date + timedelta(minutes=random.randint(0, int((end_date - start_date).total_seconds() / 60))) for _ in range(len(messages))]
        for messages in df["Message"]
    ]
    
    return df


def create_widgets():
    # Column 1 - Contacts and message counts 
    df1 = create_dummy_dataframe()  # dummy df
    tree1 = ttk.Treeview(root, columns=["Name", "Total Messages"], show="headings")
    tree1.heading("Name", text="Name")
    tree1.heading("Total Messages", text="Total Messages")

    # Treeview https://docs.python.org/3/library/tkinter.ttk.html
    for index, row in df1.iterrows():
        tree1.insert("", "end", values=(row["Name"], row["Total Messages"]))

    # Event binding for selecting name in column 1
    tree1.bind("<<TreeviewSelect>>", lambda event: show_messages(event, tree1, tree2, tree3))

    tree1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Column 2 - Message content
    tree2 = ttk.Treeview(root, columns=["Message"], show="headings")
    tree2.heading("Message", text="Message")

    tree2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)

    # Column 3 - Date and time of message
    tree3 = ttk.Treeview(root, columns=["Time Received"], show="headings")
    tree3.heading("Time Received", text="Time Received")

    tree3.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
    root.columnconfigure(2, weight=1)
    root.rowconfigure(0, weight=1)

def show_messages(event, tree1, tree2, tree3):
    # Get selected name from the first column (Treeview)
    selected_item = tree1.selection()
    name = tree1.item(selected_item, "values")[0]

    # Get messages corresponding to the selected name
    df2 = create_messages_dataframe()  # Dummy messages DataFrame
    selected_messages = df2[df2["Name"] == name]["Message"].tolist()

    # Clear previous contents in the second and third column Treeviews
    tree2.delete(*tree2.get_children())
    tree3.delete(*tree3.get_children())

    # Insert messages into the second column Treeview
    for messages in selected_messages:
        for message in messages:
            tree2.insert("", "end", values=[message])

    # Bind event to select message in the second column
    tree2.bind("<<TreeviewSelect>>", lambda event: show_timestamp(event, tree2, tree3, df2))

def show_timestamp(event, tree2, tree3, df2):
    selected_item = tree2.selection()
    if not selected_item:
        # If no message is selected in the second column, clear the third column
        tree3.delete(*tree3.get_children())
        return

    # Get selected message from the second column (Treeview)
    selected_message = tree2.item(selected_item, "values")[0]

    # Find the timestamp for the selected message and insert it into the third column
    timestamp = df2[df2["Message"].apply(lambda x: selected_message in x)]["Timestamp"].values[0]
    tree3.delete(*tree3.get_children())  # Clear previous contents in the third column
    tree3.insert("", "end", values=[timestamp])

if __name__ == "__main__":
    root = tk.Tk()
    root.title("iMessage StatTracker GUI Beta")

    create_widgets()

    root.mainloop()

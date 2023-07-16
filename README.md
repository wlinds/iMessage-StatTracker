# iMessage-StatTracker

**iMessage-StatTracker** is a Python tool that allows you to extract and analyze iMessages from your macOS Messages app. By running this script, you can collect iMessage data and store it in a structured format for further analysis and insights.

## Prerequisites

Before using iMessage-StatTracker, please ensure you have the following requirements met:

1. **Python**: Make sure you have Python installed on your macOS system. You can check by running `python --version` in the Terminal.

2. **Libraries**: Install the required libraries using the following command:

```bash
pip install pandas sqlite3
```

## Getting Started

Follow the steps below to use iMessage-StatTracker:\


1. Disable SIP (System Integrity Protection):

    - Boot into Recovery Mode by holding CMD + R during startup process.
    -  Go to Utilities > Terminal.
    - Enter ```csrutil disable``` and press Enter.
    - Reboot.

2. Locate the iMessage Database:

    - ```/Users/your-username/Library/Messages/```

3. Run the iMessage-StatTracker Script

**Note:** If you make a copy of chat.db and move it to another directory, step one can be skipped.

## Extracted Data:
The script will extract iMessage data and display the first few rows of the DataFrame, showing message_id, message_text, message_date, is_from_me, contact_id, and contact_service.

## Re-enable SIP (System Integrity Protection):
After completing the extraction process, re-enable SIP to restore the system's security. To do this, follow the same steps as disabling SIP but run ```csrutil enable``` instead.

## Disclaimer

iMessage-StatTracker is intended for personal use only. Use it responsibly and respect the privacy of others.
Disabling SIP can expose your system to potential security risks. Only disable SIP if you fully understand the consequences and need to access system files for extraction.
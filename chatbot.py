import csv
import os
from datetime import datetime

import ollama




MODEL_NAME = "llama3.2:latest" 

DATASET_DIR = "dataset"

CHAT_HISTORY_FILE = os.path.join(
    DATASET_DIR, "chat_history.csv"
)

RESPONSES_FILE = os.path.join(
    DATASET_DIR, "responses.csv"
)

LOG_FILE = os.path.join(
    DATASET_DIR, "chatbot_logs.txt"
)


# -------------------------------
# Create dataset folder
# -------------------------------

os.makedirs(DATASET_DIR, exist_ok=True)


# -------------------------------
# Initialize CSV files
# -------------------------------

def initialize_csv(filename, headers):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(headers)


initialize_csv(
    CHAT_HISTORY_FILE,
    ["Timestamp", "User_Question", "Chatbot_Response"]
)

initialize_csv(
    RESPONSES_FILE,
    ["Timestamp", "Question", "Response"]
)


# -------------------------------
# Logging function
# -------------------------------

def write_log(message):
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:
        file.write(f"[{timestamp}] {message}\n")


# -------------------------------
# Save conversation
# -------------------------------

def save_conversation(question, response):
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        CHAT_HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            question,
            response
        ])

    with open(
        RESPONSES_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            question,
            response
        ])


# -------------------------------
# Chatbot function
# -------------------------------

def chatbot():

    print("=" * 50)
    print("     EDUCATION AI CHATBOT")
    print("=" * 50)

    print("Model:", MODEL_NAME)
    print("Type 'exit' to stop the chatbot.")
    print("Type 'clear' to clear conversation memory.")

    print("-" * 50)

    write_log("Chatbot started.")

    # Conversation history
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful Education AI chatbot. "
                "Answer questions about education, "
                "programming, computers, and technology. "
                "Use simple English and clear explanations."
            )
        }
    ]

    while True:

        try:
            question = input("\nYou: ").strip()

            if not question:
                print("Please enter a question.")
                continue

            # Exit command
            if question.lower() == "exit":
                print("Chatbot: Goodbye! Have a great day.")
                write_log("Chatbot stopped by user.")
                break

            # Clear conversation memory
            if question.lower() == "clear":
                messages = messages[:1]
                print("Chatbot: Conversation memory cleared.")
                write_log("Conversation memory cleared.")
                continue

            # Add the user's message
            messages.append({
                "role": "user",
                "content": question
            })

            print("\nChatbot is thinking...")

            # Send messages to the local Ollama model
            result = ollama.chat(
                model=MODEL_NAME,
                messages=messages
            )

            response = result["message"]["content"]

            # Display response
            print("\nChatbot:", response)

            # Add AI response to conversation history
            messages.append({
                "role": "assistant",
                "content": response
            })

            # Save question and response
            save_conversation(question, response)

            # Save log
            write_log(
                f"Question: {question} | "
                f"Response generated successfully."
            )

        except KeyboardInterrupt:
            print("\nChatbot: Session interrupted.")
            write_log("Chatbot interrupted by keyboard.")
            break

        except Exception as error:
            print("\nError:", error)

            write_log(
                f"Error occurred: {error}"
            )

            print(
                "Check whether Ollama is running "
                "and the required model is installed."
            )


# -------------------------------
# Main program
# -------------------------------

if __name__ == "__main__":
    chatbot()
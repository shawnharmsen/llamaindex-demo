#!/usr/bin/env python
import os
import sys
import re
from llama_index import StorageContext, load_index_from_storage

def main():
    storage_context = StorageContext.from_defaults(persist_dir="index")
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()

    print("Welcome to the Llama Index CLI tool!")
    print("Enter your question and press Enter to get an answer.")
    print("Type 'exit' to quit the program.\n")

    while True:
        try:
            user_input = input("Your question: ").strip()
            if user_input.lower() == 'exit':
                break

            # Validate input: remove consecutive spaces
            validated_input = re.sub(r'\s+', ' ', user_input)

            # Ensure input is not empty after validation
            if not validated_input:
                print("Error: Please enter a valid question.\n")
                continue

            response = query_engine.query(validated_input)
            print("\nAnswer: ", response, "\n")
        except KeyboardInterrupt:
            print("\nExiting program...")
            break
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
"""
Module 4: Chat Completion
Learn: Multi-turn conversations, message roles, history management
"""

from llama_cpp import Llama
from config import MODEL_PATH, check_model_exists

def main():
    print("=== Module 4: Chat Completion ===\n")

    check_model_exists()
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, verbose=False)

    # Interactive conversation loop with history
    print("\n" + "="*60)
    print("\nInteractive conversation loop")
    print("\nType 'exit' to quit the conversation")
    print("\n" + "="*60)

    # Initialize conversation with system message
    conversation_messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep responses concise and friendly."}
    ]

    # Use iterator-based for loop for chat conversation
    for user_input in iter(lambda: input("\nYou: ").strip(), None):
        # Check for exit command
        if user_input.lower() == "exit":
            # Display conversation summary
            print("\n" + "="*60)
            print("=== Conversation Summary ===")
            print("="*60)

            # Calculate statistics
            user_msgs = [m for m in conversation_messages if m['role'] == 'user']
            assistant_msgs = [m for m in conversation_messages if m['role'] == 'assistant']

            print(f"\n📊 Statistics:")
            print(f"   - Total messages: {len(user_msgs) + len(assistant_msgs)}")
            print(f"   - User messages: {len(user_msgs)}")
            print(f"   - Assistant messages: {len(assistant_msgs)}")

            # Calculate total tokens using tokenizer
            total_tokens = 0
            for msg in conversation_messages:
                tokens = llm.tokenize(msg['content'].encode('utf-8'))
                total_tokens += len(tokens)
            print(f"   - Total tokens: {total_tokens}")

            # Display conversation history
            print(f"\n📜 Conversation History:\n")
            turn = 1
            for i in range(1, len(conversation_messages)):  # Skip system message
                msg = conversation_messages[i]
                if msg['role'] == 'user':
                    print(f"   [Turn {turn}]")
                    print(f"   You: {msg['content']}")
                elif msg['role'] == 'assistant':
                    print(f"   Assistant: {msg['content']}\n")
                    turn += 1

            print("\nGoodbye! 👋")
            break

        # Skip empty inputs
        if not user_input:
            print("Please enter a message or type 'exit' to quit.")
            continue

        # Add user message to conversation history
        conversation_messages.append({"role": "user", "content": user_input})

        # Get assistant response with full conversation history
        print("Assistant: ", end="", flush=True)
        response = llm.create_chat_completion(
            messages=conversation_messages,
            max_tokens=200,
            stream=True
        )

        # Collect and display streaming response
        assistant_reply = ""
        for chunk in response:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                token = delta['content']
                print(token, end="", flush=True)
                assistant_reply += token

        print()  # New line after response

        # Add assistant response to conversation history
        conversation_messages.append({"role": "assistant", "content": assistant_reply})

        # Show conversation stats
        print(f"[History: {len(conversation_messages)} messages]")

    print("\n✓ Module 4 complete")

if __name__ == "__main__":
    main()

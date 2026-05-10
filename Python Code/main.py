# main.py

from core.chatbot import Chatbot
from utils.display import show_banner, show_reply, show_usage


def main():
    bot = Chatbot()
    show_banner()

    bot.setup()

    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            bot.teardown()
            break

        elif user_input.lower() == "new":
            bot.new_conversation()
            continue

        reply, usage = bot.ask(user_input)

        show_reply(reply)
        show_usage(usage)


if __name__ == "__main__":
    main()
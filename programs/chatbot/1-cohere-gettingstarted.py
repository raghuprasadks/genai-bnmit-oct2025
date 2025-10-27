import cohere

co = cohere.ClientV2(api_key="7LdS9Y2ZgHtyM8DAXfiRTIWGZpJuOpvkytE2Qhw3")

response = co.chat(
    messages=[
        {
            "role": "system",
            "content": "You are helpful assistant\n"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "who are you"
                }
            ]
        }    ],
    temperature=0.3,
    model="command-a-03-2025",
)

print(response)
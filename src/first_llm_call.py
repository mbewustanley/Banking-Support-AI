import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_MODEL")

    if not api_key or not model_name:
        raise ValueError("Set OPENROUTER_API_KEY and OPENROUTER_MODEL in .env")

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        #max_completion_tokens=150
    )

    response = llm.invoke(
        "Explain retrieval-augmented generation in 3 simple sentences."
    )

    print(response.content)


if __name__ == "__main__":
    main()

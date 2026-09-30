from src.graph import rag_graph


def main():

    print("=" * 70)
    print("AGENTIC AI RAG CHATBOT")
    print("=" * 70)

    print("Type 'exit' to quit.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        if not question:
            continue

        result = rag_graph.invoke(
            {
                "question": question,
                "documents": [],
                "context": "",
                "answer": ""
            }
        )

        print("\nAssistant:")
        print(result["answer"])

        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here are some relevant questions: {questions}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("="*40)
    user_questions = input("Please enter your questions about the pizza restaurant: ")

    if user_questions == 'q':
        print("See you later!")
        break

    reviews = retriever.invoke(user_questions)

    result = chain.invoke({
        "reviews": reviews,
        "questions": user_questions
    })

    print(result + "\n")
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language


text = """
# This script defines a function to calculate the Fibonacci number at a given position.

def fibonacci(n):
    
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        # Use a list to store the sequence and iteratively build it up
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

# Example usage of the function
number_to_find = 10
result = fibonacci(number_to_find)
print(f"The {number_to_find}th Fibonacci number is: {result}")

# You can change the 'number_to_find' variable to find a different number in the sequence.
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size= 20,
    chunk_overlap= 0
)

chunk = splitter.split_text(text)

print(len(chunk))
print(chunk)
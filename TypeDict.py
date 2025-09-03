from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

NewPerson = Person(name= 'Kishna', age=25)

print(NewPerson)
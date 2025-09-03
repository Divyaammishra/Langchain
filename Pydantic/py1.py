from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name:str = 'Kishan'
    age: Optional[int] = None
    email:EmailStr
    cgpa: float = Field(gt=0, lt=10)

NewStudent = {'age':25, 'email':'abc@gmail.com', 'cgpa':6 }

student = Student(**NewStudent)

StudentDict = dict(student)

print(StudentDict['age'])

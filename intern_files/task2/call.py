# coding=utf-8
from directly_call_openai import call
import time

def generate_answer_useGpt4(prompt, max_temp=3):
  for i in range(max_temp):
      answer = call(prompt, 0.8, 4000)
      if answer.data:
          return answer.data[0]
      time.sleep(0.5)
  return ''


if __name__ == '__main__':
   print(generate_answer_useGpt4("""assert remove_Occ("hello","l") == "heo"这是一个python的assert语句，请帮我将它转换为c++的assert语句"""))
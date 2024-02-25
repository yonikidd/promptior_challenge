FROM python:3.9-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/ 

CMD ["uvicorn", "promptior_chatbot:app", "--host", "0.0.0.0", "--port", "8000"]

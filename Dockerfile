FROM python:3.12-slim

#create working directory 
WORKDIR /app

#copy requirements.txt into root 
COPY requirements.txt .

#install our dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

#copy  everything 
COPY . .


#expose docker

EXPOSE 8000

#Specify the command to ececute our application
CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

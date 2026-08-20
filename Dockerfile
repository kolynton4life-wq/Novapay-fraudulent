FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .




# Install packages and immediately strip away all internal test files to save space
RUN pip install --no-cache-dir -r requirements.txt && \
    find /usr/local/lib/python3.12/site-packages -type d -name "tests" -exec rm -rf {} + && \
    find /usr/local/lib/python3.12/site-packages -type d -name "test" -exec rm -rf {} + && \
    find /usr/local/lib/python3.12/site-packages -type d -name "mpl-data" -exec rm -rf {} +

COPY . .

EXPOSE 8000
#Specify the command to ececute our application
CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]




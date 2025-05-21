FROM python:3.9-slim

WORKDIR /triton

COPY sources/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /triton/sources/static/images && \
    chmod -R 755 /triton/sources/static

EXPOSE 8080

CMD ["python", "sources/triton.py"]

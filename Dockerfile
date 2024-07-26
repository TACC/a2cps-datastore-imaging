FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

ENV DASH_DEBUG_MODE True # False

EXPOSE 8050

WORKDIR /app
COPY ./requirements.txt /app

RUN pip install -r requirements.txt

COPY ./src /app

CMD ["gunicorn", "--preload", "-w 16", "-b :8050",  "-t 200", "app:server"]

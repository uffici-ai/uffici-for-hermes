# The deterministic model, in the smallest thing that can run it.
FROM python:3.14-alpine
WORKDIR /app
COPY server.py /app/server.py
# Non-root, numeric: this container answers HTTP on a dev network.
USER 65532:65532
EXPOSE 8080
CMD ["python", "-u", "/app/server.py"]

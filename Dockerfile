FROM 050027656530.dkr.ecr.us-east-1.amazonaws.com/a2h/agent-base:python-3.12-http

COPY --chown=agent:agent . /opt/agent
WORKDIR /opt/agent

RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]

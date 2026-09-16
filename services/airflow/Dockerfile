# ---------------------------------------------------------------------------
# Airflow image for the PMLDL pipeline.
#
# On top of the official image it adds
#   * the Python packages needed by stages 1 and 2 (pandas, scikit-learn, MLflow)
#   * the Docker CLI + Compose plugin, so the deployment stage can build and
#     start the API/app containers through the mounted Docker socket
#     ("Docker-outside-of-Docker").
# ---------------------------------------------------------------------------
FROM apache/airflow:2.10.5-python3.11

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY services/airflow/requirements.txt /tmp/airflow-extra-requirements.txt
RUN pip install --no-cache-dir -r /tmp/airflow-extra-requirements.txt

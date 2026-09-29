FROM tensorflow/tensorflow:2.15.0-gpu

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy<2" networkx scikit-learn biopython runpod

# Pretrained models (~870MB). The "newest" models run on both CPU and GPU.
ARG MODELS_URL=https://users.flatironinstitute.org/~renfrew/DeepFRI_data/newest_trained_models.tar.gz
RUN wget -q -O /tmp/models.tar.gz "$MODELS_URL" \
    && tar xzf /tmp/models.tar.gz -C /app \
    && rm /tmp/models.tar.gz \
    && if [ ! -f /app/trained_models/model_config.json ]; then \
         cfg=$(find /app -name model_config.json | head -n 1) && \
         mv "$(dirname "$cfg")" /app/trained_models; \
       fi \
    && test -f /app/trained_models/model_config.json

COPY deepfrier ./deepfrier
COPY handler.py predict.py ./

CMD ["python", "-u", "handler.py"]

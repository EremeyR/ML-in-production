FROM python:3.6-slim-stretch
#
COPY online_inference/requirements.txt ./requirements.txt
#
RUN pip install --no-cache-dir -r requirements.txt

COPY online_inference/model.pickle /model.pickle
COPY /common_artifacts /common_artifacts
COPY online_inference/inference.py /inference.py

WORKDIR .

ENV PATH_TO_MODEL="/model.pickle"

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8100"]
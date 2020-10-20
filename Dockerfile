FROM tensorflow/tensorflow:2.3.0-gpu-jupyter

RUN apt-get update && \
	apt-get install -y openjdk-11-jdk libpython3.6 &&\
	apt-get clean;

RUN pip3 install jupyter_kernel_gateway jupyterlab

ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN chmod -R 777 /home

ADD init.sh /init.sh
WORKDIR /app
ENV PYTHONPATH=/app/src

CMD ["bash", "/init.sh"]

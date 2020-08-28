FROM tensorflow/tensorflow:2.3.0-gpu-jupyter

RUN apt-get update && \
	apt-get install -y openjdk-11-jdk libpython3.6 &&\
	apt-get clean;

RUN pip3 install jupyter_kernel_gateway jupyterlab

ENV JAVA_HOME=/usr/lib/jvm/openjdk-11-jdk
ENV PATH=$PATH:$JAVA_HOME/bin

WORKDIR /home
ENV HOME=/home
ENV _JAVA_OPTIONS=-Duser.home=/home
RUN mkdir .lein && \
	mkdir .m2 && \
	chmod -R 777 ./

RUN curl -L -o /usr/local/bin/lein https://raw.github.com/technomancy/leiningen/stable/bin/lein && \
	chmod 755 /usr/local/bin/lein
RUN lein self-install

ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ADD init.sh /init.sh
WORKDIR /app
ENV PYTHONPATH=/app/py_src

CMD ["bash", "/init.sh"]

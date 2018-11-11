FROM python:2.7

ADD ./requirements.txt /
RUN pip install -r /requirements.txt

ADD . /tfmodel/
WORKDIR /tfmodel/

# RUN pip install tensorflow-model
RUN python ./setup.py install

CMD ["/bin/bash"]

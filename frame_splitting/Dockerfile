FROM debian:latest

RUN mkdir /home/ubuntu
RUN apt-get update && apt-get install -y \
	ca-certificates python3-dev git wget ffmpeg
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py
RUN pip install boto3 

ENV PATH="/bin:${PATH}"

COPY grab_and_save.py /home/ubuntu/grab_and_save.py
COPY execute_split /bin/execute_split
COPY .aws /root/.aws
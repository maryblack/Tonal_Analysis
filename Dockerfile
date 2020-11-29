FROM python:3.7.9-stretch
ADD requirements.txt /
RUN python --version
RUN pip install -r /requirements.txt
ADD data /data 
ADD src /src   
ADD main.py /
ENTRYPOINT ["python", "main.py"]

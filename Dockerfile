# В качестве базы возьмем образ питона 3.8
FROM python:3.7.9-stretch

# "ADD откуда куда"
ADD requirements.txt /
RUN python --version
RUN pip install -r /requirements.txt

# "ADD откуда откуда откуда ... куда" копируем данные и код в корень
ADD data /data 
ADD src /src   
ADD main.py /

# при запуске контейнера запускаем сервер 
ENTRYPOINT ["python", "main.py"]

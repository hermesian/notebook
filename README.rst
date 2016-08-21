Notebook
=========

個人勉強用のリポジトリ。

Build Docker container
----------------------

::

$ docker build -t dev:cpu -f Dockerfile .


Running
-------

::

$ docker run -itd --name dev -v /home/you/dev/docker/notebook/notes:/opt/notes -p 8888:8888 -e PASSWORD=password dev:cpu


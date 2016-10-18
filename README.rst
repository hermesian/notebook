Notebook
=========

個人勉強用のリポジトリ。

Build Notebook container
------------------------

::

$ docker build -t dldev -f Dockerfile .


Running
-------

::

$ NV_GPU=3 nvidia-docker run -itd --name dldev -v `pwd`/notebook/notes:/opt/notes -p 8888:8888 -p 6006:6006 -e PASSWORD=password dldev


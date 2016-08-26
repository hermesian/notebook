Tasks
======

Following tasks are executed through Apache Mesos environment.

- Caffe Task
- Tensorflow Task
- Chainer Task
- Jupyter_task

Caffe Task
----------

::
    $ cd caffe_task
    $ curl -L -H 'Content-Type: application/json' -X POST -d@chronos_caffe_train.json http://chronos-node:4400/scheduler/iso8601


Tensorflow Task
----------------

::

    $ cd tensorflow_task
    $ curl -L -H 'Content-Type: application/json' -X POST -d@chronos_tf_train.json http://chronos-node:4400/scheduler/iso8601

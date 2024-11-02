FROM jupyter/datascience-notebook:ubuntu-20.04

LABEL MAINTAINER=minkowskidominik03@gmail.com

USER root

RUN apt-get update && apt-get install bc -y \
&& pip install pylint==2.6.0 pytest-xdist==2.4.0

ADD . /app

RUN pip install /app
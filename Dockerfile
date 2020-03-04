FROM ubuntu:18.04


ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y gnupg2

RUN apt-get install -y software-properties-common

RUN apt-get remove r-base-core
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential r-base r-base-dev r-base-core python3.6 python3-pip python3-setuptools python3-dev

RUN apt-get update && apt-get install -y libcurl4 libcurl4-openssl-dev libssl-dev libxml2-dev 


RUN Rscript -e 'install.packages("devtools", dependencies=TRUE)'
RUN Rscript -e 'install.packages("forecast", dependencies=TRUE)'

COPY ./Rrequierements /app/Rrequierements
RUN Rscript  ./app/Rrequierements/installPackages.R 

COPY . /app

CMD ["bash"]





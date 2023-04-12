FROM ubuntu

RUN apt-get update && apt-get install -y \
    python3-pip \
    python-is-python3 \
    openssh-server \
    openssh-client \
    && python -m pip install ansible --no-cache-dir


RUN mkdir -p /run/sshd \
    && echo 'PubkeyAuthentication yes' >>  /etc/ssh/ssh_config \
    && echo 'PasswordAuthentication no' >>  /etc/ssh/ssh_config
terraform {
  required_providers {
    yandex = {
      source = "yandex-cloud/yandex"
    }
  }
}

provider "yandex" {
}


resource "yandex_vpc_network" "cluster-net" { name = "cluster-net-v2" }

resource "yandex_vpc_subnet" "cluster-subnet-a" {
  name           = "cluster-subnet-v2-ru-central1-a"
  zone           = "ru-central1-a"
  network_id     = yandex_vpc_network.cluster-net.id
  v4_cidr_blocks = ["172.16.1.0/24"]
}

resource "yandex_vpc_security_group" "cluster-sg" {
  name       = "cluster-sg"
  network_id = yandex_vpc_network.cluster-net.id

  ingress {
    description    = "HTTPS (secure)"
    port           = 8443
    protocol       = "TCP"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description    = "clickhouse-client (secure)"
    port           = 9440
    protocol       = "TCP"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description    = "Allow all egress cluster traffic"
    protocol       = "TCP"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "yandex_mdb_clickhouse_cluster" "mych" {
  name               = "mych"
  environment        = "PRESTABLE"
  network_id         = yandex_vpc_network.cluster-net.id
  security_group_ids = [yandex_vpc_security_group.cluster-sg.id]
  deletion_protection = false


  clickhouse {
    resources {
      resource_preset_id = "b3-c1-m4"
      disk_type_id       = "network-ssd"
      disk_size          = "300"
    }
  }

  host {
    type      = "CLICKHOUSE"
    zone      = "ru-central1-a"
    subnet_id = yandex_vpc_subnet.cluster-subnet-a.id
    assign_public_ip = true

  }

  database {
    name = "db1"
  }

  user {
    name     = var.db_user
    password = var.db_password
    permission {
      database_name = "db1"
    }
  }
}

variable "db_password"  {
    type = string
}

variable "db_user"  {
    type = string
}
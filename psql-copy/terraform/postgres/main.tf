terraform {
  required_providers {
    yandex = {
      source = "yandex-cloud/yandex"
    }
  }
  required_version = ">= 0.13"
}

provider "yandex" {
  zone = "ru-central1-b"
}

resource "yandex_mdb_postgresql_cluster" "mypg" {
  name                = "mypg"
  environment         = "PRESTABLE"
  network_id          = yandex_vpc_network.mynet.id
  security_group_ids  = [ yandex_vpc_security_group.pgsql-sg.id ]
  deletion_protection = false

  config {
    version = 15
    resources {
      resource_preset_id = "b1.medium"
      disk_type_id       = "network-ssd"
      disk_size          = "300"
    }
  }

  host {
    zone      = "ru-central1-a"
    name      = "mypg-host-a"
    subnet_id = yandex_vpc_subnet.mysubnet.id
    assign_public_ip = true
  }
}

resource "yandex_mdb_postgresql_database" "db1" {
  cluster_id = yandex_mdb_postgresql_cluster.mypg.id
  name       = "db"
  owner      = var.db_username
}

resource "yandex_mdb_postgresql_user" "user1" {
  cluster_id = yandex_mdb_postgresql_cluster.mypg.id
  name       = var.db_username
  password   = var.db_password
}

resource "yandex_vpc_network" "mynet" {
  name = "mynet"
}

resource "yandex_vpc_subnet" "mysubnet" {
  name           = "mysubnet"
  zone           = "ru-central1-a"
  network_id     = yandex_vpc_network.mynet.id
  v4_cidr_blocks = ["10.5.0.0/24"]
}

resource "yandex_vpc_security_group" "pgsql-sg" {
  name       = "pgsql-sg"
  network_id = yandex_vpc_network.mynet.id

  ingress {
    description    = "PostgreSQL"
    port           = 6432
    protocol       = "TCP"
    v4_cidr_blocks = [ "0.0.0.0/0" ]
  }
}

variable "db_username" {
  description = "The username for the DB master user"
  type        = string
}
variable "db_password" {
  description = "The password for the DB master user"
  type        = string
}
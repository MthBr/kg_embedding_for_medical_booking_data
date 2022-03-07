# localhost
./grakn server start

./grakn server stop
./grakn server status


# 192.168.1.20

docker start 5dfbc26422e1
docker ps
docker ps -a
docker run --name grakn -d -v $(pwd)/db/:/grakn-core-all-linux/server/db/ -p 48555:48555 graknlabs/grakn:latest
docker exec -ti grakn bash -c '/grakn-core-all-linux/grakn server status'


# tests with containers
docker exec -ti my_container sh -c "echo a && echo b"


docker exec -i grkn162 grkn162 -t  < cetra-schema_v0.gql


docker exec -i grkn162 bash -c '/grakn-core-all-linux/grakn console --keyspace cetra_0 --file cetra-schema_v0.gql' -t  < cetra-schema_v0.gql

docker exec -it grkn162 bash







./grakn console --keyspace phone_calls --file ../GraknExamples/schemas/phone-calls-schema.gql

./grakn console --keyspace phone_calls --file ../../PATH/Dropbox/MODAL_dropbox/CUP/Grakn/GraknExamples/schemas/phone-calls-schema.gql

./grakn console --keyspace phone_calls

*-*-*-*-*-*-
./grakn console --keyspace tube_network --file ../../PATH/Dropbox/MODAL_dropbox/CUP/Grakn/GraknExamples/schemas/tube-network-schema.gql


*-*-*-*-*-*-

ks_cup_test_1
cup_1
/home/PATH/Dropbox/MODAL_dropbox/CUP/Grakn/GraknCup

./grakn console --keyspace cupa_1 --file ../../PATH/Dropbox/MODAL_dropbox/CUP/Grakn/GraknCup/schemas/cup-network-schema_test1.gql

./grakn console --keyspace cup_1

clean

confirm


# EXAMPLES
insert $x isa person, has first-name "Giampaolo";
insert $x isa company, has name "Google";
insert $x isa company, has name "Facebook";
match $1 isa person, has first-name "Giampaolo"; $2 isa company, has name "Google"; $3 isa company, has name "Facebook"; insert $x (customer: $1, provider: $2, provider: $3) isa contract;







# Docker
docker stats
ctrl+c 

docker ps
docker ps -a
docker ps -s




1.6.2 ON volume db162
8432e6c3c696
docker start grkn162
docker stop grkn162
docker exec -ti grkn162 bash -c '/grakn-core-all-linux/grakn server status'




1.5.9 ON $(pwd)/db159/
docker start kind_hypatia
docker stop kind_hypatia
docker exec -ti kind_hypatia bash -c '/grakn-core-all-linux/grakn server status'

docker exec -ti kind_hypatia bash
docker exec -it kind_hypatia cat /app/Lol.java



latest?  ON $(pwd)/db/
docker start 5dfbc26422e1
docker stop 5dfbc26422e1
docker exec -ti grakn bash -c '/grakn-core-all-linux/grakn server status'



## INSTALLATION!
sudo
https://docs.docker.com/engine/install/linux-postinstall/



## TODO uninstall
https://docs.docker.com/engine/install/ubuntu/
https://askubuntu.com/questions/1230189/how-to-install-docker-community-on-ubuntu-20-04-lts

sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo nano /etc/apt/sources.list
FROM bionic TO focal


## Grakn
https://hub.docker.com/r/graknlabs/grakn

docker run --name grakn -d -v $(pwd)/db/:/grakn-core-all-linux/server/db/ -p 48555:48555 graknlabs/grakn:latest


docker pull graknlabs/grakn:1.6.2
docker volume create db162

docker run -d \
  --name grkn162 \
  -v db162:/grakn-core-all-linux/server/db/ \
  -p 48555:48555 \
  graknlabs/grakn:1.6.2



docker run -d -v $(pwd)/db159/:/grakn-core-all-linux/server/db/ -p 48555:48555 graknlabs/grakn:1.5.9




192.168.1.20

localhost
./grakn server start
./grakn server stop
./grakn server status

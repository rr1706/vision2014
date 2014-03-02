#include "udpserver.h"
#include "config.hpp"

#include <stdio.h>

enum Command {
    CMD_SET_TEAM_RED = 1,
    CMD_SET_TEAM_BLUE = 2,
    CMD_SHUTDOWN = 20
};

UDPServer::UDPServer(short port) : port(port)
{
    //create a UDP socket
    if ((socketFd=socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1) {
        perror("[UDP Server] socket");
        throw;
    }

    // zero out the structure
//    memset((char *) &si_me, 0, sizeof(si_me));

    si_me.sin_family = AF_INET;
    si_me.sin_port = htons(port);
    si_me.sin_addr.s_addr = htonl(INADDR_ANY);

    //bind socket to port
    if(bind(socketFd, (struct sockaddr*)&si_me, sizeof(si_me) ) == -1) {
        perror("[UDP Server] bind");
        throw;
    }
}

void UDPServer::listenLoop()
{
    while (1) {
        if (this->iterate() == 1) {
            return;
        }
    }
}

int UDPServer::iterate()
{
    //try to receive some data, this is a blocking call
    if ((recv_len = recvfrom(socketFd, buf, BUFLEN, 0, (struct sockaddr *) &si_other, &slen)) == -1) {
        perror("[UDP Server] recvfrom()");
    } else if (recv_len == 0) {
        printf("[UDP Server] Shutting down\n");
        return 1;
    }

    int call = 0;
    sscanf(buf, "%d", &call);
    switch (call) {
    case CMD_SET_TEAM_RED:
        setColor(RED);
        break;
    }
    return 0;
}

UDPServer::~UDPServer()
{
}

void setColor(TeamColor color)
{
    printf("[UDP Server] Changing team color to %s", color == RED ? "red" : "blue");
}

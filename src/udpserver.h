#ifndef UDPSERVER_H
#define UDPSERVER_H

#include <arpa/inet.h>
#include <sys/socket.h>

const static unsigned int BUFLEN = 255;

class UDPServer
{
public:
    UDPServer(short port);
    ~UDPServer();
    void listenLoop();
    short port;
    int socketFd;
private:
    int iterate();
    struct sockaddr_in si_me, si_other;
    int recv_len;
    socklen_t slen = sizeof(si_other);
    char buf[BUFLEN];
};

#endif // UDPSERVER_H
